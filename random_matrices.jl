export random_pure, random_overlap, random_unitary, random_max_ent, random_product
using CUDA, LinearAlgebra, TensorCast

CUDA.allowscalar(false)

function my_kron!(z::CuMatrix, x::CuMatrix, y::CuMatrix)
    function kernel(z, x, y)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i <= size(z, 2)
            d1 = size(x, 1)
            d2 = size(y, 1)
            for k=1:d1, l=1:d2
                z[(k-1)*d2+l, i] = x[k, i] * y[l, i]
            end
        end
        return
    end
    @assert size(z, 2) == size(x, 2) == size(y, 2)
    @assert size(z, 1) == size(x, 1) * size(y, 1)
    threads = 512
    blocks = cld(size(x, 2), threads)
    @cuda threads=threads blocks=blocks kernel(z, x, y)
end

function gram_schmidt_step(x, y)
    d, batchsize = size(x)
    overlaps = CUDA.zeros(eltype(x), 1, batchsize)
    # <y, x> x
    conj!(x)
    CUDA.CUBLAS.gemv_strided_batched!('N', 1.0, reshape(y, 1, d, batchsize), x, 1, overlaps)
    conj!(x)
    proj = overlaps .* x
    return y - proj
end

function random_pure(::Type{T}, d::Int, batchsize::Int) where {T}
    ψd = CUDA.randn(T, d, batchsize)
    norm_invs = T.(1 ./ sqrt.(sum(abs.(ψd) .^ 2, dims = 1)))
    ψd = ψd .* norm_invs
    return ψd
end

function random_product(::Type{T}, d::Int, batchsize::Int) where {T}
    local_d = isqrt(d)
    ψd = random_pure(T, local_d, batchsize)
    ϕd = random_pure(T, local_d, batchsize)
    ξd = CUDA.zeros(T, d, batchsize)
    my_kron!(ξd, ψd, ϕd)
    return ξd
end

function random_overlap(::Type{T}, d, batchsize, q::Real) where {T}
    x = random_pure(T, d, batchsize)
    y = random_pure(T, d, batchsize)
    xp = gram_schmidt_step(x, y)
    return sqrt(1 - q^2) * xp + q * x, x
end

function random_unitary(::Type{T}, d, batchsize) where {T}
    g = CUDA.randn(T, d, d, batchsize)
    _tau, _r = CUBLAS.geqrf_batched!([view(g, :, :, i) for i=1:batchsize])
    r = CUDA.zeros(T, d, d, batchsize)
    tau = CUDA.zeros(T, d, batchsize)
    correction_factors = CUDA.zeros(T, d, batchsize)
    CUDA.@time @views Threads.@threads for i=1:batchsize
        correction_factors[:, i] = sign.(diag(_r[i]))
        _r[i][diagind(_r[i])] .= 1
        r[:, :, i] = tril(_r[i])
        tau[:, i] = _tau[i]
    end
    id = CuArray(I, d, d)
    @cast b[k, l, i, m] := id - r[k, i, m] * conj(r[l, i, m]) * tau[i, m]
    Q = CUDA.zeros(T, d, d, batchsize)
    CUDA.@time @views Threads.@threads for i=1:batchsize
        Q[:, :, i] = b[:, :, 1, i] * b[:, :, 2, i]
        for j=3:size(b,  3)
            Q[:, :, i] *= b[:, :, j, i]
        end
    end
    @cast Q[i, j, k] = Q[i, j, k] * correction_factors[i, k]
    return Q
end

function random_unitary_batch(T::Type, d, batchsize)
    Qs = CuArray{T, 3}(undef, d, d, batchsize)  
    for i in 1:batchsize
        A = CUDA.randn(T, d, d)   
        F = qr(A)                 
        Q = Matrix(F.Q)           
        Q .*= sign(det(Q))        
        Qs[:,:,i] .= CuArray(Q)    
    end
    return Qs
end
function random_max_ent(::Type{T}, d, batchsize) where {T}
    Q = random_unitary(T, d, batchsize)
    ψ = reduce(hcat, vec.(eachslice(Q; dims=3))) ./ d
    return ψ
end

function random_unitary_cpu(::Type{T}, d, batchsize) where {T}
    Qs = Vector{Matrix{T}}(undef, batchsize)
    for i in 1:batchsize
        g = randn(T, d, d)
        Qs[i], _ = qr(g)
    end
    return Qs
end
function random_unitary_gpu(::Type{T}, d, batchsize) where {T}
    A = CUDA.randn(T, d, d, batchsize)
    Q = CUDA.zeros(T, d, d, batchsize)
    R = CUDA.zeros(T, d, d, batchsize)

  CUDA.@time @views Threads.@threads for i in 1:batchsize
    Ai = A[:, :, i]
    F = CUDA.qr(Ai)
    Qi = CuArray(F.Q)   
    Ri = F.R           

    phases = sign.(diag(Ri))

    Qi_scaled = Qi .* reshape(phases, 1, :)
    copyto!(view(Q, :, :, i), Qi_scaled)
    copyto!(view(R, :, :, i), Ri)
end

    return Q, R
end
function test(::Type{T}, d, batchsize) where {T}
    A = CUDA.randn(T, d, d, batchsize)   
    Q = CUDA.zeros(T, d, d, batchsize)   
    R = CUDA.zeros(T, d, d, batchsize)  

    CUDA.@time @views Threads.@threads for i in 1:batchsize
        Ai = view(A, :, :, i)
        F = qr(Ai) 
        Qi = F.Q
        Ri = F.R

    
        phases = CUDA.sign.(diag(Ri))        
        Q[:, :, i] .= Qi .* phases'            
        R[:, :, i] .= Ri
    end

    return Q, R
end


d = 10
batchsize = 10000
n_iter = 5
times = Float64[]

# for _ in 1:n_iter
#     GC.gc()  # opcjonalnie wymusza garbage collection
#     t0 = time()
#     Q = test(ComplexF32, d, batchsize)
#     t1 = time()
#     push!(times, t1 - t0)
# end
println("Czasy: ", times)
times = Float64[]
for _ in 1:n_iter
    GC.gc()  # opcjonalnie wymusza garbage collection
    t0 = time()
    Q = random_unitary_gpu(ComplexF32, d, batchsize)
    t1 = time()
    push!(times, t1 - t0)
end

println("Czasy: ", times)
# println("Średni czas: ", mean(times))
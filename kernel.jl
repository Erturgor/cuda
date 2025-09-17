using CUDA, LinearAlgebra, TensorCast

CUDA.allowscalar(false)
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
using JSON
using LinearAlgebra

struct QKDProtocol 
    name::String
    qa::Vector{Float64}
    rhos::Vector{Vector{ComplexF64}}
    povms::Vector{Vector{ComplexF64}}
    f::Tuple{Vector{Float64}, Vector{Vector{Int}}}

    function QKDProtocol(name::String, qa::Vector{Float64}, list_rho::Vector{Vector{ComplexF64}},
                         povm::Vector{Vector{ComplexF64}}, f::Tuple{Vector{Float64}, Vector{Vector{Int}}})
        if !isapprox(sum(qa), 1.0; atol=1e-8)
            error("qa must sum to 1")
        end
        if !isapprox(sum(f[1]), 1.0; atol=1e-8)
            error("f[1] must sum to 1")
        end
        dim = length(list_rho[1])
        total = zeros(ComplexF64, dim, dim)
        for x in povm
            total += x * x'
        end
        if !isapprox(total, I, atol=1e-8)
            error("POVM does not sum to identity")
        end

        new(name, qa, list_rho, povm, f)
    end
end



function complex_to_jsonvec(vecs::Vector{Vector{ComplexF64}})
    [ [ [real(x), imag(x)] for x in v ] for v in vecs ]
end


function json_to_complexvec(vecs)
    [ [ ComplexF64(x[1], x[2]) for x in v ] for v in vecs ]
end

function save_protocol(protocol::QKDProtocol, filename::String)
    data = Dict(
        "name" => protocol.name,
        "qa" => protocol.qa,
        "rhos" => complex_to_jsonvec(protocol.rhos),
        "povms" => complex_to_jsonvec(protocol.povms),
        "f" => protocol.f
    )
    open(filename, "w") do io
        JSON.print(io, data)
    end
end

function load_protocol(filename::String)
    data = JSON.parsefile(filename)

    rhos = json_to_complexvec(data["rhos"])
    povms = json_to_complexvec(data["povms"])

    f1 = Float64.(data["f"][1])
    f2 = [Int.(pair) for pair in data["f"][2]]

    return QKDProtocol(
        data["name"],
        Float64.(data["qa"]),
        rhos,
        povms,
        (f1, f2)
    )
end


qkd = QKDProtocol(
    "BB84",
    [1/4, 1/4, 1/4, 1/4],
    [
        [1+0im, 0+0im],
        [0+0im, 1+0im],
        [1+0im, 1+0im]/sqrt(2),
        [1+0im, -1+0im]/sqrt(2)
    ],
    [
        [0+0im, 1+0im]/sqrt(2),
        [1+0im, 0+0im]/sqrt(2),
        [1+0im, -1+0im]/2,
        [1+0im, 1+0im]/2
    ],
    ([1/2, 1/2], [[1, 2], [3, 4]])
)

# --- Save and load ---
save_protocol(qkd, "qkd.json")
protocol2 = load_protocol("qkd.json")

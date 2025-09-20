using Convex, SCS
using LinearAlgebra
using QuantumInformation
using Combinatorics
using SparseArrays
using Plots
const MOI = Convex.MOI
function y1_sdp(ϵ, qa, list_rho, povm, f)
    
    dim = size(list_rho[1],1)
    d2 = dim^2


    z = Variable()
    J_Phi = ComplexVariable(d2, d2)
    J_Phi_star = ComplexVariable(d2, d2)
    Y = ComplexVariable(d2,d2)

    # J(Id)
    Ω = vec(Matrix(I,dim,dim))
    J_Id = Ω*Ω'

   constraints = [
        0 <= z, z <= ϵ,
        
        J_Phi in :SDP,
        partialtrace(J_Phi, 1, [dim, dim]) == I(dim),
        
        J_Phi_star in :SDP,
        partialtrace(J_Phi_star, 1, [dim, dim]) == I(dim),
        
        J_Phi == (1 - z) * J_Id + Y,

        # The LMI trick to enforce Y == z * J_Phi_star
        #J_Phi == (1 - z)*J_Id + z*J_Phi_star cant work on Convex
        [z * Matrix(I, d2, d2) Y; Y' J_Phi_star] in :SDP
    ]

    # P(A=B|S)
    total = 0.0
    S = length(f[1])
    for s in 1:S
        s_prob = f[1][s]
        for a_idx in f[2][s]
            #Because in QKD we choose povm to be orthogonal i had to switch here?
            b_idx = (a_idx % 2 == 0) ? (a_idx - 1) : (a_idx + 1)
            a_prob = qa[a_idx]
            rho_a = list_rho[a_idx]
            M_b = povm[b_idx]
            op = kron(M_b, transpose(rho_a))

            push!(constraints, real(tr(J_Phi * op)) >= 0)

            total += s_prob * a_prob * real(tr(J_Phi * op))
        end
    end

    problem = minimize(total, constraints)
    solve!(problem, MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-6, "eps_rel" => 1e-6); silent_solver=true)
    return problem.optval
end


function y2_sdp(ϵ, qa, list_rho, povm, f)
    dim = size(list_rho[1],1)
    d2 = dim^2
    J_Phi = ComplexVariable(d2, d2)

    Ω = vec(Matrix(I, dim, dim))
    J_Id = Ω * Ω'

    constraints = []

    push!(constraints, J_Phi in :SDP) 

    push!(constraints, partialtrace(J_Phi, 1, [dim, dim]) == I(dim))

    # Dual Diamond norm constraint
    Y = Semidefinite(d2)
    LMI = [ Y     J_Phi - J_Id;
            (J_Phi - J_Id)'        Y ]
    push!(constraints, LMI in :SDP)
    push!(constraints, tr(Y) <= 2*ϵ)


    total = 0.0
    S = length(f[1])
    for s in 1:S
        s_prob = f[1][s]
        for a_idx in f[2][s]
            b_idx = (a_idx % 2 == 0) ? (a_idx - 1) : (a_idx + 1)
            a_prob = qa[a_idx]
            rho_a = list_rho[a_idx]
            M_b = povm[b_idx]
            op = kron(M_b, transpose(rho_a))
            total += s_prob * a_prob * real(tr(J_Phi * op))
            # push!(constraints, real(tr(J_Phi * op)) >= 0)
        end
    end

    push!(constraints, total > 0)

    problem = minimize(total, constraints)
    solve!(problem,
           MOI.OptimizerWithAttributes(SCS.Optimizer,
                                    "eps_abs"=>1e-10,
                                   "eps_rel"=>1e-10)
                                   ,
           silent_solver=true)

    return problem.optval
end


function SDP(qa, list_rho, povm, f, y)
    """
      
        qa - distribution of Alice
        list_rho - list of states
        povm - Bob's POVM
        f - decision function (distribution, points)
    """

    S = length(f[1])

    # Eve (BI_in ⊗ BI_out), (e ⊗ s), Eve[e][s]
    Eve = [[ComplexVariable(4,4) for _=1:S] for _=1:2]
    constraints = vec([Eve[e][s] in :SDP for e=1:2, s=1:S])
    for s=1:(S-1)
        constraints += [sum(Eve[e][s] for e=1:2) == sum(Eve[e][s+1] for e=1:2)]
    end
    constraints += [partialtrace(sum(Eve[e][s] for e=1:2, s=1:S), 2, [2,2]) == sum(tr(Eve[e][s]) for e=1:2, s=1:S)*I(2)/2]

    cond_prob(e,b,a,s) = tr(
        Eve[e][s] * (conj.(list_rho[a]) ⊗ povm[b])
    )

    # Probs
    PS = real(sum(f[1][s] * sum(qa[a]*cond_prob(e,b,a,s) for e=1:2, a=f[2][s], b=f[2][s]) for s=1:S))
    PB = real(sum(f[1][s] * sum(qa[a]*cond_prob(e,b,a,s) for e=1:2, a=f[2][s], b=f[2][s] if a!=b) for s=1:S))
    PE = real(sum(f[1][s] * sum(qa[f[2][s][i]]*cond_prob(i,b,f[2][s][i],s) for i=1:2, b=f[2][s]) for s=1:S))
    
    # SDP
    constraints += [PS==1]
    constraints += [PB>=y]
    problem = maximize(PE, constraints)
    
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )
    print("$(problem.status): $(problem.optval) \n")
    return problem.optval
end

function y_sdp_combined(ϵ, qa, list_rho, povm, f)

    #y1 = y1_sdp(ϵ,qa, list_rho, povm, f)
    #println("y1 = $y1")
    y2 = y2_sdp(ϵ, qa, list_rho, povm, f)
    #result_y1 = SDP(qa, list_rho, povm, f,y1)
    
    result_y2 = SDP(qa, list_rho, povm, f,y2)
    println("y2 = $y2")
    #return min(result_y1, result_y2)
    return result_y2
end

function results(methods)
    ϵ = 0.0
    interval = (ϵ):0.01:(1.0)

    outs = [[y_sdp_combined(ϵ, method[1]...) for ϵ in interval] for method in methods]
    return interval, outs
end

function save_and_draw(methods, file)
    interval, outs = results(methods)

    Pl=plot(interval, interval, color="black",  aspect_ratio=1, label = "\$\\mathbb{P}(B=A|S)\$" ,legend=:outertopright )
    for i in eachindex(methods)
        plot!(Pl, interval, outs[i], color = palette(:tab10)[mod(i-1,10)+1], label = "\$\\mathbb{P}(E=A|S)\$ [$(methods[i][2])]")
    end

    xlabel!("\$x\$", fontsize=20)
    ylabel!("\$\\mathbb{P}\$", fontsize=20)
    savefig(Pl,"$(file).pdf")
end

function BB84()
    name = "BB84"
    qa = [1/4, 1/4, 1/4, 1/4]
    list_rho = [[1, 0], [0,1] , [1,1]/sqrt(2), [1,-1]/sqrt(2)]
    povm = [[0,1]/sqrt(2), [1,0]/sqrt(2), [1,-1]/2, [1,1]/2]
    f = [[1/2, 1/2],[[1,2], [3,4]]]
    return [qa, [x*x' for x in list_rho], [x*x' for x in povm], f], name
end
function SARG04()
    name = "SARG04"
    qa = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    list_rho = [[1, 0], [0,1] , [1,1]/sqrt(2), [1,-1]/sqrt(2), [1,im]/sqrt(2), [1,-im]/sqrt(2)]
    povm = [[0,1]/sqrt(3), [1,0]/sqrt(3), [1,-1]/sqrt(6), [1,1]/sqrt(6), [1,-im]/sqrt(6), [1,im]/sqrt(6)]
    f = [[1/3, 1/3, 1/3],[[1,2], [3,4], [5,6]]]
    return [qa, [x*x' for x in list_rho], [x*x' for x in povm], f], name
end
#save_and_draw( [BB84()], "test")

function find_epsilon_max_grid(qa, list_rho, povm, f; step=0.01)
    ϵ_values = 0.0:step:1.0

    results = []

    for ϵ in ϵ_values

        y1 = y1_sdp(ϵ, qa, list_rho, povm, f)
        println("y1 = $y1")
        y2 = y2_sdp(ϵ, qa, list_rho, povm, f)
        println("y2 = $y2")

        result_y1 = SDP(qa, list_rho, povm, f, y1)
        println("result_y1 = $result_y1")
        result_y2 = SDP(qa, list_rho, povm, f, y2)
        println("result_y2 = $result_y2")


        result_y = min(result_y1, result_y2)

        push!(results, (ϵ, result_y))
    end
    ϵ_max_found = maximum([ϵ for (ϵ, ry) in results if ry > 0])
    return ϵ_max_found
end

data, name = SARG04()
qa, list_rho, povm, f = data

ϵ_max = find_epsilon_max_grid(qa, list_rho, povm, f)

println("ϵ_max = $ϵ_max")
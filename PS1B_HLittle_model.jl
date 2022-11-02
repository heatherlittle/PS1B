using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, PrettyTables #import the libraries we want
#note that StatFiles allows us to load a stata .dta StatFiles

#navigate to the correct folder, I assume we're in /Users/hlittle
#cd("/Users/hlittle/Desktop/PS1B")

@with_kw struct Primitives
#as online, load data: https://juliapackages.com/p/statfiles
df = DataFrame(load("/Users/hlittle/Desktop/PS1B/Mortgage_performance_data.dta")) 
mat_df = Matrix(df) #turn the data frame into a Matrix

#concatenate the columns associated with noted vectors into your matrix X, we assume that the last bit is a list, not a difference
#X = Array{Float64, 2}
X::Array{Float64, 2} = hcat(ones(16355), mat_df[:,24], mat_df[:,25], mat_df[:,5], mat_df[:,21], mat_df[:,3], mat_df[:,7], mat_df[:,8], mat_df[:,9], mat_df[:,10], mat_df[:,14], mat_df[:,16], mat_df[:,23], mat_df[:,27], mat_df[:,28], mat_df[:,29], mat_df[:,30])
#call the outcome variable, i_close_first_year (20th column)
#Y = Array{Float64, 1}
Y::Array{Float64, 1} = mat_df[:, 20]

#beta as instructed for part 1
#β::Array{Float64, 1} 
#β = zeros(17)
#β[1] = -1

end #close Primitives struct

function logistic(x)
    y = exp(x)/(1+exp(x))
    return y
end

function Log_Like(beta::Array{Float64, 1}, x::Matrix{Float64}, y::Vector{Float64})

    N = length(y) #so we can loop over each observation

    sum = 0 #initialize
    for i = 1:N
        sum += log(logistic(dot(x[i,:], beta))^(y[i])*(1-logistic(dot(x[i,:], beta)))^(1-y[i]))
    end #close the for loop

    return sum

end #close function for log likelihood

function Log_Like_b(beta::Array{Float64, 1})

    N = length(prim.Y) #so we can loop over each observation

    sum = 0 #initialize
    for i = 1:N
        sum += log(logistic(dot(prim.X[i,:], beta))^(prim.Y[i])*(1-logistic(dot(prim.X[i,:], beta)))^(1-prim.Y[i]))
    end #close the for loop

    return -sum #in addition to making this only be a function of beta, I put the negative sign here since we're maximizing, not minimizing

end #close function for log likelihood

function Score(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    N = length(y) #so we can loop over each observation
    K = length(x[1,:]) #so we can initialize the score

    sum = zeros(K) #the score is a 1xK length vector
    for i = 1:N
        sum += (y[i]-logistic(dot(x[i,:], beta))).*x[i,:]
    end #close the for loop

    return sum

end #close function for score

function Hessian(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    N = length(y) #so we can loop over each observation
    K = length(x[1,:]) #so we can initialize the hessian

    sum = zeros(K,K) #the Hessian is a KxK matrix
    for i = 1:N
        X_mat = x[i,:]*x[i,:]'
        scale = logistic(dot(x[i,:], beta))*(1-logistic(dot(x[i,:], beta)))
        sum += scale.*X_mat
    end #close the for loop

    return -sum #note that the Hessian is the negative of the sum we've constructed
    
end #close hessian function

function numeric_FDeriv(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    #generate the log likelihood using unperturbed beta
    LL_orig = Log_Like(beta, x, y)

    #initialize a gradient
    K = length(x[1,:]) #so we can initialize the gradient
    gradient = zeros(K)
    pert = 0.00001
    for i=1:K
        beta_p = beta
        beta_p[i] = beta[i] + pert #perturbing just the one element of beta at a time
        LL_pert = Log_Like(beta_p, x, y)
        gradient[i] = (LL_pert - LL_orig)/pert
    end #close loop perturbing each beta value

    return gradient

end #close numerical first order derivative function

function numeric_SDeriv(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    #generate the log likelihood using unperturbed beta
    LL_orig = Log_Like(beta, x, y)

    #initialize a hessian
    K = length(x[1,:]) #so we can initialize the gradient
    hess = zeros(K,K)
    pert = 0.00001

    for i = 1:K
        #construct perturbation along rows
        beta_p_i = beta
        beta_p_i[i] = beta[i] + pert
        LL_pert_i = Log_Like(beta_p_i, x, y)

        for j = 1:K
            #construct perturbation along columns
            beta_p_j = beta
            beta_p_j[j] = beta[j] + pert
            LL_pert_j = Log_Like(beta_p_j, x, y)

            #fill in the hessian
            hess[i, j] = (LL_pert_i + LL_pert_j - 2*LL_orig)/(pert^2)
        end #close loop over j
    end #close loop over i

    return hess

end #close numerical second order derivative function

function Newton_Solve(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    #because we will be updating b and b_next in the loop, start by iterating them outside the loop
    b = beta
    b_next = zeros(length(beta))

    #initialize an error term and tolerance
    error = 100.0
    tol = 0.00000000001

    n = 0 #counter
    while error>tol
        b_next = b -inv(Hessian(b, x, y))*Score(b, x, y) #see JF's slide 8/24, we don't augement by line search
        diff = abs.(b_next.-b) #this an next line are like two step sup norm
        error = maximum(diff) #update the error term

        #if the error is big enough to continue, update b with the b_next to proceed
        b = b_next
        n += 1 #update the counter

    end #end the while loop

    println("The Newton Algorithm took ", n, " iterations to converge. Our estimate for beta is ", b_next, ".")

    return b_next

end #close the function to solve ML problem with a newton algorithm









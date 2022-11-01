using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames #import the libraries we want
#note that StatFiles allows us to load a stata .dta StatFiles

#navigate to the correct folder, I assume we're in /Users/hlittle
#cd("/Users/hlittle/Desktop/PS1B")

#do this in the compute file
#=
#initialize the data
df = DataFrame(load("/Users/hlittle/Desktop/PS1B/Mortgage_performance_data.dta")) #as online, load data: https://juliapackages.com/p/statfiles
mat_df = Matrix(df) #turn the data frame into a Matrix

#concatenate the columns associated with noted vectors into your matrix X, we assume that the last bit is a list, not a difference
#X = Array{Float64, 2}
X::Array{Float64, 2} = hcat(ones(16355), mat_df[:,24], mat_df[:,25], mat_df[:,5], mat_df[:,21], mat_df[:,3], mat_df[:,7], mat_df[:,8], mat_df[:,9], mat_df[:,10], mat_df[:,14], mat_df[:,16], mat_df[:,23], mat_df[:,27], mat_df[:,28], mat_df[:,29], mat_df[:,30])
#call the outcome variable, i_close_first_year (20th column)
#Y = Array{Float64, 1}
Y::Array{Float64, 1} = mat_df[:, 20]
=#

function logistic(x)
    y = 1/(1+exp(x))
    return y
end

function Log_Like(beta::Array{Float64, 1}, x::Matrix{Float64}, y::Vector{Float64})

    N = length(y) #so we can loop over each observation

    sum = 0 #initialize
    for i = 1:N
        sum += log(logistic(x[i,:]*beta)^(y[i])*(1-logistic(x[i,:]*beta))^(1-y[i]))
    end #close the for loop

    return sum

end #close function for log likelihood

function Score(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    N = length(y) #so we can loop over each observation
    K = length(x[1,:]) #so we can initialize the score

    sum = zeros(K) #the score is a 1xK length vector
    for i = 1:N
        sum += (y[i]-logistic(x[i,:]*beta)).*x[i,:]
    end #close the for loop

    return sum

end #close

function Hessian(beta::Array{Float64, 1}, x::Array{Float64, 2}, y::Array{Float64, 1})

    N = length(y) #so we can loop over each observation
    K = length(x[1,:]) #so we can initialize the hessian

    sum = zeros(K,K) #the Hessian is a KxK matrix
    for i = 1:N
        sum += logistic(x[i,:]*beta)*(1-logistic(x[i,:]*beta)).*(x[i,:]'*x[i,:])
    end #close the for loop

    return -sum #note that the Hessian is the negative of the sum we've constructed
    
end







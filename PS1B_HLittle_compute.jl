using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames #import the libraries we want

include("PS1B_HLittle_model.jl") #import the functions we've written

#No primitive struct

##############################################################################################
#initialize the data
##############################################################################################
#as online, load data: https://juliapackages.com/p/statfiles
df = DataFrame(load("/Users/hlittle/Desktop/PS1B/Mortgage_performance_data.dta")) 
mat_df = Matrix(df) #turn the data frame into a Matrix

#concatenate the columns associated with noted vectors into your matrix X, we assume that the last bit is a list, not a difference
#X = Array{Float64, 2}
X::Array{Float64, 2} = hcat(ones(16355), mat_df[:,24], mat_df[:,25], mat_df[:,5], mat_df[:,21], mat_df[:,3], mat_df[:,7], mat_df[:,8], mat_df[:,9], mat_df[:,10], mat_df[:,14], mat_df[:,16], mat_df[:,23], mat_df[:,27], mat_df[:,28], mat_df[:,29], mat_df[:,30])
#call the outcome variable, i_close_first_year (20th column)
#Y = Array{Float64, 1}
Y::Array{Float64, 1} = mat_df[:, 20]
##############################################################################################


##############################################################################################
#for part 1, we'll feed in the specified beta vector to the log likelihood, score, and hessian
##############################################################################################

#create the specified vector of beta
β = zeros(17)
β[1] = -1

#use the functions and X and Y from the model file
Log_Likelihood1 = Log_Like(β, X, Y)
Score1 = Score(β, X, Y)
Hessian1 = Hessian(β, X, Y)






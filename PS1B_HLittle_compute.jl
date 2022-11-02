using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames, Optim, PrettyTables #import the libraries we want
#note that I don't use PrettyTables, but this is where I learn about how we can export nice looking tables in latext code

include("PS1B_HLittle_model.jl") #import the functions we've written

prim = Primitives()

β = zeros(17)
β[1] = -1

##############################################################################################
#for part 1, we'll feed in the specified beta vector to the log likelihood, score, and hessian
##############################################################################################

#use the functions and X and Y from the model file
Log_Likelihood1 = Log_Like(β, prim.X, prim.Y)
Log_Likelihood_test = Log_Like_b(β)
Score1 = Score(β, prim.X, prim.Y)
Hessian1 = Hessian(β, prim.X, prim.Y)


##############################################################################################
#for part 2, we calculate the numerical derivative
##############################################################################################

#the numerical first order derivative
numeric_FDeriv(β, prim.X, prim.Y)

numeric_SDeriv(β, prim.X, prim.Y)


##############################################################################################
#for part 3, solve the maximum likelihood problem using the newton algorithm
##############################################################################################

b_guess = Newton_Solve(β, prim.X, prim.Y)
println(b_guess)
#saving this will allow us to use it as an initial guess below!

##############################################################################################
#for part 4, use the optimization package for BFGS and Simplex (note Nelder Mead is default)
##############################################################################################

b_BFGS = optimize(Log_Like_b, b_guess, BFGS()) #note that this took 390 seconds to run the first time (I did not provide gradient)
println("The vector that minimizes with the BFGS algorithm is ", b_BFGS.minimizer, ".")

b_Simplex = optimize(Log_Like_b, b_guess) #note that the default, Nelder Mead, is the simplex method; this took 25 seconds when I ran it
println("The vector that minimizes with the Simplex algorithm is ", b_Simplex.minimizer, ".")



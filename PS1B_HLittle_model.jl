using Parameters, LinearAlgebra, Distributions, Random, Statistics, StatFiles, DataFrames #import the libraries we want
#note that StatFiles allows us to load a stata .dta StatFiles

#navigate to the correct folder, I assume we're in /Users/hlittle
#cd("/Users/hlittle/Desktop/PS1B")

#as online, load data: https://juliapackages.com/p/statfiles
df = DataFrame(load("/Users/hlittle/Desktop/PS1B/Mortgage_performace_data.dta"))

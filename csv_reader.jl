include("ESN.jl")



df = DataFrame( CSV.File("MRESN_tanh_mnist_GPU_IT.csv") )
ttl= "MRESN"

for n in names(df)
    println(n)
end

df = df[1:75,["Alpha", "Initial transient", "Error"]]

df


describe(df)

unique(df,:Alpha)



using StatsPlots

x = [i for i in 1:10]'
y = hcat([le[i].Error for i in 1:10 ]...)

StatsPlots.boxplot(x, y, xlabel="Number of layers", ylabel="Error", title=ttl, legend=false, alpha=0.7)
include("../ESN.jl")

dir = "../data/cryptocurrency/"

fls = readdir(dir)

dfs = DataFrame.(CSV.File.( [dir*f for f in fls] ) )


ij = innerjoin(dfs... , on = :Date; makeunique=true)

names = DataFrame(select(ij, r"^Name")[1,:])

sij = select(ij, r"^[Open,Close]")

CSV.write(dir*"unified_open_close.csv", sij)

CSV.write(dir*"unified_open_close_names.csv", names)

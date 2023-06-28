# import Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
using CSV;
using DataFrames;

file = "DATASET02.csv"

df = DataFrame(CSV.File(file))

c2 = df[:,2]
c3 = df[:,3]

df = []

result_file = open("turbine_column_2.txt", "a")
for c in c2
    s = string(c)*"\n"
    write(result_file, s)
end
close(result_file)

result_file = open("turbine_column_3.txt", "a")
for c in c3
    s = string(c)*"\n"
    write(result_file, s)
end
close(result_file)


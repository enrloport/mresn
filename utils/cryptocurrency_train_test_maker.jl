include("../ESN.jl")

currency= "Monero"
dir     = "../data/cryptocurrency/"

df      = DataFrame.(CSV.File.( dir*"unified_open_close.csv" ) )
df_names= DataFrame.(CSV.File.( dir*"unified_open_close_names.csv" ) )
names   = Dict(
    df_names[1,df_name] => string(df_name)
    for df_name in propertynames(df_names)
)


target  = names[currency]
col     = replace(target, "Name_" => "Close_")

train   = df[1:end-1, Not(col)]
test    = df[2:end,col]

function make_train_test_csv()
    CSV.write(dir*"train_"*currency*".csv", train; header=false)
    CSV.write(dir*"test_"*currency*".csv", Tables.table(test); header=false)
end

make_train_test_csv()
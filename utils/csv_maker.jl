include("ESN.jl")

file_tr = "data/satelite/cloudcast/TrainCloud.nc"
file_te = "data/satelite/cloudcast/TestCloud.nc"
ncinfo(file_tr)

 _data = ncread(file_tr, "__xarray_dataarray_variable__")
#_data = ncread(file_te, "__xarray_dataarray_variable__")

_all   = mapslices(x-> Float64.(x) ,_data, dims=(2,3))
_first = _all[1,:,:]'
maximum(_first)

__data = mapslices(x->imresize(x, (40,40)), _all, dims=(2,3) )
data_n = mapslices(x-> x./255, __data, dims=(2,3))

target = (22,33)
labels = data_n[:,target...]
m      = maximum(labels)

n,w,h = size(data_n)


ae = AdaptiveEqualization(nbins = 256, rblocks = 4, cblocks = 4, clip = 0.2)
eq_tr0 = mapslices(x -> adjust_histogram(x, ae) , data_n, dims=(2,3))

eq_vec = reshape(eq_tr0, n, h*w )

t=900
display(Images.Gray.( hcat(eq_tr0[t,:,:]' ,reshape(eq_vec[t,:,:], h,w)') ) )

size(eq_vec)

eq_file = "data/satelite/cloudcast/eq_40x40.csv"
# eq_file = "eq_40x40.csv"
CSV.write(eq_file,  Tables.table(eq_vec), writeheader=false)

eq_vec = CSV.read("data/satelite/cloudcast/eq_40x40.csv", DataFrame; header=false)

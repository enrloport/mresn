include("ESN.jl")

_dir = "../data/cloudcast/"
file_t = _dir*"TrainCloud.nc"
#file_t = _dir*"TestCloud.nc"
ncinfo(file_t)

_data = ncread(file_t, "__xarray_dataarray_variable__")
_data = map(x-> x % Int, _data)

n,w,h = size(_data)

radius = 3

for i in 1+radius:w-radius, j in 1+radius:h-radius

    # Maximum real value is 14. Only non valid pixels have a value of 254/255
    eq_vec = map(x-> x > 253 ? 0 : x, _data[:, i-radius:radius+i, j-radius:radius+j])
    rshped = reshape(eq_vec, :, (2*radius + 1)^2 ) 
    
    dir_res = "../data/cloudcast/splitted_radius_3/"
    _name = "cloudcast_EQ_radius-" * string(radius) * "_(" * string(i) *","* string(j) *").csv"
    CSV.write(dir_res*_name,  Tables.table(rshped), writeheader=false)
    
end

# using ImageView
# Images.ImageShow(data_int , axes=(2,3))
function __time_now()
    return string(Dates.today()) *"_"* string( Dates.Time(Dates.now()))
end


function set_spectral_radius!(W::Mtx, rho::Float64=1.0)
    lambda  = maximum(abs.(eigvals(W)))
    W     .*= (rho / lambda)
    return W
end


#=
    Given a point in a 2d matrix (x,y) with dimensions max_x * max_y, it returns the 
    moore neighborhood of the point. No toroidal conexions are considered.
=#
function moore_neighborhood(x,y, max_x, max_y)
    ng = [
         (x-1, y-1), (x-1, y), (x-1, y+1)
        ,(x  , y-1), (x  , y), (x  , y+1)
        ,(x+1, y-1), (x+1, y), (x+1, y+1)    
    ]
    valid_ng = [x for x in ng if 0 < x[1] <= max_x && 0 < x[2] <= max_y ]
    return valid_ng
end


#=
    Given a reservoir R, a node and a list of neighbors, it modify the reservoir by creating 
    edges between the node and its neighbors. 
    Two flavours: 
        Use set_edges_value! if you want to give same value to the edges.
        Use set_edges_rand! if you want to give a random uniform value to the edges. Sigma param will set the range of the distribution.
=#
function set_edges_value!(R::Mtx, node::Int, neighs::Array{Int} ; value::Float64=0.5, final_size=(0,0))
    max    = maximum([node, neighs...])
    
    for neigh in neighs
        R[node, neigh] = value
    end
    if final_size != (0,0)
        res = zeros(final_size...)
        m,n = size(R)
        res[1:m, 1:n] = R
        return res
    end
    return R
end

function set_edges_rand!(R::Mtx, node::Int, neighs::Array{Int}; sigma=1.0, final_size=(0,0))
    max    = maximum([node, neighs...])
    
    for neigh in neighs
        R[node, neigh] = rand(Uniform(-sigma, sigma))
    end
    if final_size != (0,0)
        res = zeros(final_size...)
        m,n = size(R)
        res[1:m, 1:n] = R
        return res
    end
    return R
end


function increase_R(R, incr, new_edges=0; sparse=false, R_scaling=1 )
        
    size_R = size(R)[1]
    A      = zeros(size_R+incr, size_R+incr)
    A[1:size_R, 1:size_R] = R

    if new_edges > 0
        range_1 = range( size_R+1, size_R + incr)
        range_2 = range( 1       , size_R + incr)

        set = Set()
        for i in range_1, j in range_2
            push!(set, (i,j))
            push!(set, (j,i))
        end
        new_positions = shuffle(collect(set))[1:new_edges]

        for ij in new_positions
            A[ij[1], ij[2]] = rand(Uniform(-R_scaling,R_scaling))
        end
    end

    if sparse return sparse(A) end    
    return A
end


function __pop!(list, index)
    chosen  = list[index]
    deleteat!(list,index)
    return chosen
end


function break_path!(R, pos)
    R[ pos... ] = 0.0
    return R
end

function set_link!(R::Mtx, position::Tuple, value)
    R[position...] = value
    return R
end


function unweighted_digraph_R(R::Mtx)
    links = map( x-> isnan(x) ? 0.0 : x, R./R )
    return SimpleDiGraph(links)
end


# Given an image and a raius of pixels, this function applies blur to the image using the pixel radius.
function blur_img(image, n)
    reshape([
        __blur_px(image, x,y, n)
        for x in 1:size(image)[1], y in 1:size(image)[2]
    ], size(image))
end

function __blur_px(image, x,y, n)
    mean(
        __get_pixel(image, i, j)
        for i in x-n:x+n, j in y-n:y+n
    )
end

function __get_pixel(image, x,y)
    w,h = size(image)
    # mirror over top/left
    x,y = abs(x-1)+1, abs(y-1)+1
    # mirror over the bottom/right
    if x > w
        x = w+ (w - x)
    end
    if y > h
        y = h+(h - y)
    end
    return image[x, y]
end

# Given an image (Matrix) and the limits of the new range, it changes the range of pixel intensity values.
function linear_normalization!(x, new_min=0.0, new_max=1.0)
    min, max = minimum(x), maximum(x)
    factor   = (new_max-new_min)/(max-min)
    x        = ((x .- min).* factor) .+ new_min
    return x
end

# Given a train set and a test set, this functions returns a dictionary with blured copies of original datasets.
# It also normalize image in a specific range (0.0, 1.0 by default) and resize the image (Original image size is used by default)
function data_blur(tr_x_original, te_x_original; size=size(tr_x_original,1), min=0.0, max=1.0, steps=2 )
    tr_x = tr_x_original
    te_x = te_x_original
    sz = size

    train_x = mapslices(x -> imresize(linear_normalization!(x', min, max), (sz,sz)) , tr_x, dims=(1,2) )
    test_x  = mapslices(x -> imresize(linear_normalization!(x', min, max), (sz,sz)) , te_x, dims=(1,2) )

    train_x_blur = Dict( i => mapslices( x-> reshape(linear_normalization!(blur_img(x, i),min,max) ,:,1), train_x, dims=(1,2) ) for i in 1:steps)
    test_x_blur = Dict( i => mapslices( x-> reshape(linear_normalization!(blur_img(x, i),min,max) ,:,1), test_x, dims=(1,2) ) for i in 1:steps)

    train_x = mapslices(x -> reshape(x,:,1) , train_x, dims=(1,2) )
    test_x  = mapslices(x -> reshape(x,:,1) , test_x, dims=(1,2) )    
    
    return train_x, test_x, train_x_blur, test_x_blur
end


# Useful for MLDatasets
# Given an array of classes, a tensor of images (images in dimensions 1 and 2) and an array of labels, this function
# returns a dictionary where the keys are the labels of each class and the values are the list of instances belonging to that class.
function make_classes_dict(classes::Array{Int}, datax::Array{T}, datay::Array{Int}) where T<:AbstractFloat
    dict = Dict( 
        c => reshape( reduce(hcat,[datax[:,:,i] for i in 1:length(datay) if datay[i] == c]) ,size(datax,1),size(datax,2),: )
        for c in classes
    )
    return dict
end

# Useful for MLDatasets
# Given an array of target classes and a dictionary of classes, this function returns an array with the instances of the
# dictionary that belongs to target classes and an array of labels for each instance.
function data_filter(targets::Vector{Int}, t_dict::Dict{Int,Array{T,3}}; dim=3) where T<:AbstractFloat
    datax = t_dict[targets[1]]
    datay = floor.(Int, ones(size(t_dict[targets[1]],dim)) .* targets[1])

    for i in 2:lastindex(targets)
        datax = cat(datax, t_dict[targets[i]], dims=dim )
        datay = vcat(datay, floor.(Int, ones(size(t_dict[targets[i]],dim)) .* targets[i]) )
    end
    return datax, datay
end


# Given a tested ESN, and the number of classes, this functions returns the confusion matrix of the given ESN.
function confusion_matrix(tested_reservoir, number_of_classes::Int) 
    res, nc = tested_reservoir, number_of_classes - 1
    pd = Dict( (x,y) => 0 for x in 0:nc, y in 0:nc )
    for pairs in res[:wrong_class][:, :]
        sp = [pairs[2][2],pairs[2][3]]
        pd[ (sp[1],sp[2]) ] += 1
    end

    nc += 1
    cm = zeros(nc,nc)
    for i in 1:nc, j in 1:nc
        cm[i,j] = pd[(i-1,j-1)] 
    end

    return cm
end

# Decide if the ecuations in matrix X have an independient term
function __new_column(u, x, zero_degree)
    res = zero_degree ? [u;x;1] : [u;x]
    return res
end




# This function is used by  __do_train_image_forecast! function.
# Input:
#   - tr_data is the Input tensor transformed into a matrix (each row contains an image represented as a vector)
#   - X is the training matrix when we have an asociation betwen an image and a state
#   - beta is the classic hiperparameter of esn
#   - min is the number of steps of the train tensor that we have to skip (initial_transient + forecasting_horizon)
# Result:
#   - It returns a matrix of dimensions number_of_pixels x number_of_pixels + number_of_nodes
#     each row of the result is a Wout computed adhoc for each pixel.
function __train_wouts(tr_data,X,beta,min)
    n_px = size(tr_data,2)
    res = zeros( n_px, size(X,1) )
    for pix in 1:n_px
        target = pix
        data   = tr_data[:,target]
        res[pix:pix,:] = Matrix( transpose( (X*transpose(X) + beta*I)\(X*data[min:end]) ) )
    end
    return res
end

function __train_wouts_GPU(tr_data,X,beta,min)
    n_px = size(tr_data,2)
    res = CuArray(zeros( n_px, size(X,1) ))
    for pix in 1:n_px
        target = pix
        data   = CuArray(tr_data[:,target])
        res[pix:pix,:] = Matrix( transpose( (X*transpose(X) + beta*I)\(X*data[min:end]) ) )
    end
    return res
end



# This function updates the state of the neurons. No feedback.
function __update_x!(Win,W,x,u,alpha)
    x[:] = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x)
end


# This function fills the matrix Y using CuArrays.
# Inputs:
#   - args: Dictionary with main parameters.
#   - x: state Matrix
#   - u: input
#   - Y: the matrix to be filled.
function __fill_Y_GPU!(args,x,u,Y)
    for t in 1:args[:horizon] # Last examples of training
        u[:] = CuArray(reshape(args[:train_data][end-(args[:horizon]-t),:,:], :, 1))
        __update_x!(args[:R_in],args[:R],x,u,args[:alpha])
        Y[:,t] = args[:result][:R_out]*args[:new_column](u,x)
    end
    for t = 1+args[:horizon]:args[:result][:test_length]
        u[:] = CuArray(reshape(args[:test_data][t-1,:,:], :, 1))
        __update_x!(args[:R_in],args[:R],x,u,args[:alpha])
        Y[:,t] = args[:result][:R_out]*args[:new_column](u,x)
    end
end



function cloudcast_to_csv(input_file, output_size)
    file = input_file
    sz   = output_size

    _data  = ncread(file, "__xarray_dataarray_variable__")
    __all  = mapslices(x-> Float64.(x) ,_data, dims=(2,3))
    __data = mapslices(x->imresize(x, (sz,sz)), __all, dims=(2,3) )
    data_n = mapslices(x-> x./255, __data, dims=(2,3))

    ae        = AdaptiveEqualization(nbins = 256, rblocks = 4, cblocks = 4, clip = 0.2)
    equalized = mapslices(x -> adjust_histogram(x, ae) , data_n, dims=(2,3))
    eq        = reshape(equalized, size(equalized,1), sz*sz )

    name = file*"__"*string(sz)*"x"*string(sz)
    CSV.write(name*".csv",  Tables.table(eq), writeheader=false)
end



# Given a radius and a pixel coordinates this function returns the limits of the subimage
function __subimage_index(rad::Int, pxl::Tuple{Int,Int})
    return pxl[1]-rad , pxl[1]+rad, pxl[2]-rad ,pxl[2]+rad
end


function mse(l1,l2)
    return sum((l1 .- l2).^2) / length(l2)
end

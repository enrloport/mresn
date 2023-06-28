include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb

# using Pkg
# Pkg.add("Noise")
using Noise


# Random.seed!(42)

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
# tr_x, tr_y = FashionMNIST(split=:train)[:]
# te_x, te_y = FashionMNIST(split=:test)[:]



function resize_map_mnist(train_x, sz, trl)
    trx = mapslices(x-> imresize(x, sz), train_x[:,:,1:trl] ,dims=(1,2))
    trx[:] = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, trx)
    return trx
end
function map_resize_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(x-> imresize(x, sz), trx[:,:,1:trl] ,dims=(1,2))
    return trx
end
px      = rand([20])
sz      = (px,px)
rm  = resize_map_mnist(test_x, sz, _params[:test_length])
mr  = map_resize_mnist(test_x, sz, _params[:test_length])


xmpl = 1
or  = test_x[:,:,xmpl]'
irm = rm[:,:,xmpl]'
imr = mr[:,:,xmpl]'

Images.Gray.(or)

display(vcat(Images.Gray.(irm), Images.Gray.(imr)))



sp = salt_pepper(test_x[:,:,2]', salt_prob= 0.5, salt=0.9, pepper=0.1)

rsp= hcat(imresize(sp, sz), zeros(20, size(train_x,1) - px ) )

Images.Gray.(vcat(sp,rsp))



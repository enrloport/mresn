################################################
include("ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
# tr_x, tr_y = FashionMNIST(split=:train)[:]
# te_x, te_y = FashionMNIST(split=:test)[:]

using Clustering

function get_centers(train_x, k, sz=(28,28))
    xr = mapslices(x-> imresize(x, sz), train_x ,dims=(1,2))
    xtr = reshape(xr, :, size(train_x, 3))

    xk = Clustering.kmeans(xtr,k)
    centers = Dict( i => reshape(xk.centers[:,i],sz[1],sz[2]) for i in 1:size(xk.centers,2) )
    display(Images.Gray.( hcat( map(x-> x', values(centers))... ) ))
    return xk, centers
end

# kmns, centers = get_centers(train_x, 10,(10,10))
# centers[2]
# display(Images.Gray.(imresize(centers[2], (10,10) )' ) )



sd = 8556
# path = "csvs/Wouts__mixed_MrESN__seed"*string(sd)*".csv"
# df = readdlm(path, ',', Float64)
# dic = Dict( (x-1) => Array(df[:,x]') for x in 1:10 )

repit = 1
sz  = (14,14)
_params = Dict{Symbol,Any}(
     :gpu           => false
    ,:wb            => false
    # ,:wout          => dic
    ,:wb_logger_name=> "mnist_mixed__MrESN"
    ,:num_esns      => 2
    ,:num_hadamard  => 2
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-6
    ,:initial_transient=>2
    ,:image_size    => sz
    ,:train_length  => 60000
    ,:test_length   => 10000
    ,:train_data    => mapslices(x-> imresize(x, sz), train_x ,dims=(1,2))
    ,:train_labels  => train_y
    ,:test_data     => mapslices(x-> imresize(x, sz), test_x ,dims=(1,2))
    ,:test_labels   => test_y
    # ,:train_f       => __do_read_Wout!
    ,:train_f       => __do_train_MrESN_mnist!
    ,:test_f        => __do_test_MrESN_mnist!
)






# function __do_read_Wout!(mrE,_params)
#     mrE.classes_Routs = _params[:wout]
#     for i in 1:length(mrE.esns)
#         mrE.esns[i].x = zeros( mrE.esns[i].R_size, 1)
#     end
#     if _params[:gpu]
#         for i in 1:length(mrE.esns)
#             mrE.esns[i].x = CuArray(mrE.esns[i].x)
#         end
#     end
# end



function do_batch(_params_esn, _params,sd)
    sz       = _params[:image_size]
    im_sz    = sz[1]*sz[2] 
    nodes    = _params_esn[:nodes] #im_sz
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    densities= _params_esn[:density]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]

    esns = [
        ESN( 
             R      = _params[:gpu] ? CuArray(new_R(nodes, density=densities[i], rho=rhos[i])) : new_R(nodes, density=densities[i], rho=rhos[i])
            ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-sigmas[i],sigmas[i]), nodes, im_sz )) : rand(Uniform(-sigmas[i],sigmas[i]), nodes, im_sz )
            ,R_scaling = r_scales[i]
            ,alpha  = alphas[i]
            ,rho    = rhos[i]
            ,sigma  = sigmas[i]
        ) for i in 1:_params[:num_esns]
    ]

    for _ in 1:_params[:num_hadamard]
        push!(esns,
            ESN(
                R      = _params[:gpu] ? CuArray(new_R_img(sz[1],sz[2])) : new_R_img(sz[1],sz[2])
                ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-0.5,0.5), 1, im_sz )) : rand(Uniform(-0.5,0.5), im_sz, 1 )
                ,hadamard=true
                ,R_scaling = 1.0
                ,alpha  = 0.7
                ,rho    = 1.0
                ,sigma  = 0.5
            )
        )
    end

    tms = @elapsed begin
        mrE = MrESN(
            esns=esns
            ,beta=_params[:beta] 
            ,train_function = _params[:train_f] #__do_read_Wout!
            ,test_function = _params[:test_f] #__do_test_MrESN_mnist!
        )
        mrE.train_function(mrE,_params)
        mrE.test_function(mrE,_params)
    end
 
    to_log = Dict(
        "Time"      => tms
       , "Error"    => mrE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
        wandb_log_artif(
            _params[:lg]
            , "Wouts__ori_MrESN__seed"*string(sd)
            , DataFrame( Dict(string(k) => vec(Array(v)) for (k,v) in pairs(mrE.classes_Routs)) )
        )
    else
        display(to_log)
    end
    return mrE
end

r1 = 0
for _ in 1:1
    # sd = rand(1:10000)
    Random.seed!(sd)
    # Random.seed!(9753)
    _params[:num_esns] = rand([5,10,15,20])
    _params[:num_hadamard]  = rand([0,1])
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
        ,:density  => rand(Uniform(0.01,0.7),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => 1000
    )
    if _params[:wb]
        using Logging
        using Wandb
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], Dict(
            "Classical reservoirs" => _params[:num_esns]
            ,"Hadamard reservoirs" => _params[:num_hadamard]
            , "Total nodes"  => _params[:num_esns] * _params_esn[:nodes] + sz[1]*sz[2] * _params[:num_hadamard]
            , "Train length" => _params[:train_length]
            , "Test length"  => _params[:test_length]
            , "seed"         => sd
            )
        )
    end
    tm = @elapsed begin
        r1 = do_batch(_params_esn,_params, sd)
    end
    if _params[:wb]
        close(_params[:lg])
    end
    println("Error: ", r1.error )
    if _params[:gpu]
        println("Time GPU: ", tm )
    else
        println("Time CPU: ", tm )
    end
end

# dump(r1)

  cm = zeros(10,10)
for i in 1:10000
    y = r1.Y[i][1]
    yt = r1.Y_target[i]
   
    if y != yt
        cm[yt+1,y+1] += 1
        # println(i," ", y, " ", yt)
    end
end

cm
sum(cm)


heatmap( 0:9, 0:9, cm, size=(310,300)  )


example = 2
println(r1.wrong_class[example][4], " ", r1.wrong_class[example][2], " ", r1.wrong_class[example][3] )
display( Images.Gray.(r1.wrong_class[example][1]') )


routs1 = r1.classes_Routs



new_classes = [5,3]
_params[:classes] = new_classes
_params[:train_x] = train_x[:,:, [ i for i in 1:_params[:train_length] if train_y[i] in new_classes ] ]
_params[:train_y] = train_y[[ i for i in 1:_params[:train_length] if train_y[i] in new_classes ] ]

_params[:test_x] = test_x[:,:, [ i for i in 1:_params[:test_length] if test_y[i] in new_classes ] ]
_params[:test_y] = test_y[[ i for i in 1:_params[:test_length] if test_y[i] in new_classes ] ]

_params[:train_length] = size(_params[:train_y],1)
_params[:test_length] = size(_params[:test_y],1)


display( Images.Gray.(_params[:test_x][:,:,1]') )

# r1.train_function = __do_train_MrESN_mnist!
# r1.train_function(r1,_params)
# r1.test_function(r1,_params)
# routs2 = r1.classes_Routs


sd = rand(1:10000)
Random.seed!(sd)
_params[:train_f] = __do_train_MrESN_mnist!
_params[:num_esns] = rand([5,10])
_params[:num_hadamard]  = rand([0,1])
_params_esn = Dict{Symbol,Any}(
    :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
    ,:density  => rand(Uniform(0.01,0.7),_params[:num_esns])
    ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:nodes    => 2000
)

_params

tm = @elapsed begin
    r2 = do_batch(_params_esn,_params, sd)
end


size(r2.wrong_class)



using Statistics
include("ESN.jl")

l = [ 0.1255,0.1233,0.1273,0.1245, 0.1244 ]

(1 - mean(l))*100
std(l)*100

l1 = string.([ 2,6,10, 15,20,30,40 ].*1000)
l2 = [ 95.6, 97.12, 97.76, 98.28, 98.36, 100-(2.49) , 100-(2.95)  ]


plt = plot(l1,l2, legend=false, linewidth=3, gridlinewidth = 2, xlabel="nodes", ylabel="Accuracy", xticks=:all)

savefig( plt, "nodes_accuracy.pdf")


########################################################

include("ESN.jl")

train_x = CSV.read("augmented_mnist_train_x.csv", DataFrame, header=false)
num_res = 20
nodes_per_res = 1000

H = zeros( num_res*nodes_per_res,size(train_x,1))




########################################################


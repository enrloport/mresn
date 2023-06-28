include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb

Random.seed!(42)

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
# tr_x, tr_y = FashionMNIST(split=:train)[:]
# te_x, te_y = FashionMNIST(split=:test)[:]

repit = 10
_params = Dict{Symbol,Any}(
     :gpu           => false
    ,:wb            => true
    ,:wb_logger_name=> "mnist_original_multilayer__MrESN"
    ,:num_esns      => 5 
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-6
    ,:initial_transient=>2
    ,:image_size    => 28*28
    ,:train_length  => 60000
    ,:test_length   => 10000
    ,:train_data    => train_x
    ,:train_labels  => train_y
    ,:test_data     => test_x
    ,:test_labels   => test_y
)


function do_batch(_params_esn, _params,sd)
    im_sz    = _params[:image_size] #28*28
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

    tms = @elapsed begin
        mrE = MrESN(
            esns=esns
            ,beta=_params[:beta] 
            ,train_function = __do_train_MrESN_mnist!
            ,test_function = __do_test_MrESN_mnist!
        )
        mrE.train_function(mrE,_params)
        mrE.test_function(mrE,_params)
    end
 
    to_log = Dict(
        "Time"          => tms
       , "Error"        => mrE.error
       # , "density"      => dens
       # , "alpha"        => _params_esn[:alpha]
       # , "rho"          => rho
       , "Train length" => _params[:train_length]
       , "Test length"  => _params[:test_length]
       #, "Nodes per reservoir"  => nodes
       , "seed"         => sd
       #, "Num ESNs"     => _params[:num_esns]
       #, "Total nodes"  => _params[:num_esns] * nodes
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
for _ in 1:repit
    sd = rand(1:10000)
    Random.seed!(sd)
    # Random.seed!(9753)
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
        ,:density  => rand(Uniform(0.01,0.7),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => 4000
    )
    if _params[:wb]
        using Logging
        using Wandb
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], Dict(
             "Nodes" => _params_esn[:nodes]
            , "Number of reservoirs" => _params[:num_esns]
            , "Total nodes"  => _params[:num_esns] * _params_esn[:nodes]
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
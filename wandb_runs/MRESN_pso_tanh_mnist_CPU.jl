include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Logging
using Wandb

############################################################################ SEED

seed = 32
# Random.seed!(seed)

############################################################################ DATASET

# MNIST dataset
train_x, train_y = MNIST(split=:train)[:]
test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
#train_x, train_y = FashionMNIST(split=:train)[:]
#test_x, test_y = FashionMNIST(split=:test)[:]


function transform_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(
        x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
    )
    return trx
end

px      = 14 # rand([14,20,25,28])
sz      = (px,px)
train_x = transform_mnist(train_x, sz, length(train_y) )
test_x  = transform_mnist(test_x, sz, length(test_y) )

# train_x = transform_mnist(train_x, sz, _params[:train_length] )
# test_x  = transform_mnist(test_x, sz, _params[:test_length])



############################################################################ PARAMETERS


repit =10
_params = Dict{Symbol,Any}(
     :gpu           => false
    ,:wb            => true
    ,:wb_logger_name=> "MRESN_pso tanh_Mnist_CPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1]
    ,:test_length   => size(test_y)[1]
    ,:train_f       => __do_train_MrESN_mnist!
    ,:test_f        => __do_test_MrESN_mnist!
    ,:num_esns      => 4
    ,:image_size   => sz
    ,:train_data   => train_x
    ,:test_data    => test_x
    ,:train_labels => train_y
    ,:test_labels  => test_y
)

_params_esn = Dict{Symbol,Any}(
    :R_scaling => [1.0 for _ in 1:_params[:num_esns] ] 
    ,:nodes    => [400 for _ in 1:_params[:num_esns] ]
)



############################################################################ LOG

par = Dict(
    "Reservoirs" => _params[:num_esns]
    , "Total nodes"        => sum(_params_esn[:nodes])
    , "Train length"       => _params[:train_length]
    , "Test length"        => _params[:test_length]
    , "Resized"            => _params[:image_size][1]
    , "Nodes per reservoir"=> _params_esn[:nodes]
    , "seed"               => seed
    , "beta"               => _params[:beta]
    , "Constant term"      => 1 # _params[:num_esns]
    , "preprocess"         => "yes"
)









############################################################################ PSO ALGORITHM


function do_batch( _params,_a,_d,_r,_s,_i )

    _params[:initial_transient] = _i

    ##### MRESN CREATION

    Random.seed!(seed) # Same seed to generate esn, but different hyperparameters due to PSO.
    esns = [
            ESN( 
                 R      = _params[:gpu] ? CuArray(new_R(_params_esn[:nodes][i], density=_d, rho=_r)) : new_R(_params_esn[:nodes][i], density=_d, rho=_r)
                ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-_s,_s), _params_esn[:nodes][i], sz[1]*sz[2] )) : rand(Uniform(-_s,_s), _params_esn[:nodes][i], sz[1]*sz[2] )
                ,R_scaling = _params_esn[:R_scaling][i]
                ,alpha  = _a
                ,rho    = _r
                ,sigma  = _s
            ) for i in 1:_params[:num_esns]
        ]


    mrE = MrESN(
        esns=esns
        ,beta=_params[:beta] 
        ,train_function = _params[:train_f]
        ,test_function = _params[:test_f]
    )

    tms = @elapsed begin

        tm_train = @elapsed begin
            mrE.train_function(mrE,_params)
        end
	    # println("Maximum X: ", maximum(mrE.X))
        println("TRAIN FINISHED, ", tm_train)
        tm_test = @elapsed begin
            mrE.test_function(mrE,_params)
        end
        println("TEST FINISHED, ", tm_test)
    end
 
    to_log = Dict(
        "Total time" => tms
        ,"Alpha" => _a
        ,"Density" =>_d
        ,"Rho" =>_r
        ,"Sigma" =>_s
        ,"Initial Transient" =>_i 
        , "Error"     => mrE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
        println(" ")
    end
    # GC.gc()
    return mrE
end


function fitness(x)
    _a,_d,_r,_s,_i = x[1], x[2], x[3], x[4], floor(x[5])

    res = do_batch(_params,_a,_d,_r,_s,_i )
    return res.error
end

pso_dict = Dict(
    "N"  => 50
    ,"C1" => 2.0
    ,"C2" => 1.5
    ,"w"  => 0.8
    ,"max_iter" => 25
)


for _it in 1:repit
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        if _it == 1
            Wandb.log(_params[:lg], pso_dict )
        end
        Wandb.log(_params[:lg], par )
    else
        display(par)
        display(pso_dict)
        println(" ")
    end


    pso = PSO(;information=Metaheuristics.Information()
        ,N  = pso_dict["N"]
        ,C1 = pso_dict["C1"]
        ,C2 = pso_dict["C2"]
        ,ω  = pso_dict["w"]
        ,options = Options(iterations=pso_dict["max_iter"])
    )

    # Cota superior e inferior de individuos. alpha, density, rho, sigma, initial transient
    lx = [0.7, 0.001, 0.001, 0.01, 1.0 ]'
    ux = [1.0,   0.7,   1.5,   1.5,  4.0 ]'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )

    if _params[:wb]
        close(_params[:lg])
    end
    GC.gc()
end

# EOF

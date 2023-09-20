include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb

############################################################################ SEED

seed = 42
Random.seed!(seed)

############################################################################ DATASET

# MNIST dataset
#train_x, train_y = MNIST(split=:train)[:]
#test_x , test_y  = MNIST(split=:test)[:]

# FashionMNIST dataset
train_x, train_y = FashionMNIST(split=:train)[:]
test_x, test_y = FashionMNIST(split=:test)[:]


function transform_mnist(train_x, sz, trl)
    trx = train_x #map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(
        x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
    )
    return trx
end

px      = 28 # rand([14,20,25,28])
sz      = (px,px)
# train_x = transform_mnist(train_x, sz, length(train_y) )
# test_x  = transform_mnist(test_x, sz, length(test_y) )

# train_x = transform_mnist(train_x, sz, _params[:train_length] )
# test_x  = transform_mnist(test_x, sz, _params[:test_length])



############################################################################ PARAMETERS


# repit =500
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => false
    ,:wb_logger_name=> "MRESN_pso glc_FashionMnist_GPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1]
    ,:test_length   => size(test_y)[1]
    ,:train_f       => __do_train_MrESN_mnist!
    ,:test_f        => __do_test_MrESN_mnist!
    ,:num_esns      => 20 # rand([10,15,20,25])
    ,:num_hadamard  => 0 # rand([1,2])
    ,:initial_transient => 1 #rand([1,2,3])
    ,:image_size   => sz
    ,:train_data   => train_x
    ,:test_data    => test_x
    ,:train_labels => train_y
    ,:test_labels  => test_y
)

min_d, max_d = 0.01, 0.2

_params_esn = Dict{Symbol,Any}(
    :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
    ,:density  => rand(Uniform(min_d, max_d),_params[:num_esns])
    ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
    ,:nodes    => [1000 for _ in 1:_params[:num_esns] ]
)



############################################################################ LOG

par = Dict(
    "Reservoirs" => _params[:num_esns]
    ,"Hadamard reservoirs" => _params[:num_hadamard]
    , "Total nodes"        => sum(_params_esn[:nodes]) + sz[1]*sz[2] * _params[:num_hadamard]
    # , "Total nodes"       => _params[:num_esns] * _params_esn[:nodes] + sz[1]*sz[2] * _params[:num_hadamard]
    , "Train length"       => _params[:train_length]
    , "Test length"        => _params[:test_length]
    , "Resized"            => _params[:image_size][1]
    , "Nodes per reservoir"=> _params_esn[:nodes]
    , "Initial transient"  => _params[:initial_transient]
    , "seed"               => seed
    , "alphas"             => _params_esn[:alpha]
    , "beta"               => _params[:beta]
    , "densities"          => _params_esn[:density]
    , "max_density"        => max_d
    , "min_density"        => min_d
    , "rhos"               => _params_esn[:rho]
    , "sigmas"             => _params_esn[:sigma]
    , "R_scalings"         => _params_esn[:R_scaling]
    , "Constant term"      => 1 # _params[:num_esns]
    , "preprocess"         => "yes"
)

if _params[:wb]
    using Logging
    using Wandb
    _params[:lg] = wandb_logger(_params[:wb_logger_name])
    Wandb.log(_params[:lg], par )
else
    display(par)
end


par = Dict(""=>0)
GC.gc()




############################################################################ MRESN CREATION

esns = [
        ESN( 
             R      = _params[:gpu] ? CuArray(new_R(_params_esn[:nodes][i], density=_params_esn[:density][i], rho=_params_esn[:rho][i])) : new_R(_params_esn[:nodes][i], density=_params_esn[:density][i], rho=_params_esn[:rho][i])
            ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-_params_esn[:sigma][i],_params_esn[:sigma][i]), _params_esn[:nodes][i], sz[1]*sz[2] )) : rand(Uniform(-_params_esn[:sigma][i],_params_esn[:sigma][i]), _params_esn[:nodes][i], sz[1]*sz[2] )
            ,R_scaling = _params_esn[:R_scaling][i]
            ,alpha  = _params_esn[:alpha][i]
            ,rho    = _params_esn[:rho][i]
            ,sigma  = _params_esn[:sigma][i]
            
        ) for i in 1:_params[:num_esns]
    ]


mrE = MrESN(
    esns=esns
    ,beta=_params[:beta] 
    ,train_function = _params[:train_f]
    ,test_function = _params[:test_f]
)




############################################################################ PSO ALGORITHM


function do_batch(mrE, _params, k,b,v,q)

    function glc(x; A=-k, K=k, C=1.0, B=b, V=v, Q=q)
        return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/V) )
    end

    tms = @elapsed begin
        for e in mrE.esns
            e.sgmd = glc
        end

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
        ,"K" => k
        ,"B" => b
        ,"V" => v
        ,"Q" => q
        , "Error"     => mrE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
        println(" ")
    end
    GC.gc()
    return mrE
end


function fitness(x)
    k,b,v,q = x[1], x[2], x[3], x[4]

    res = do_batch(mrE, _params, k,b,v,q)
    # Wandb.log(lg , Dict("error" => res[:error] ,"A" => a ,"K" => k , "B" => q, "Q" => q ) )
    return res.error
end



pso = PSO(;information=Metaheuristics.Information()
    ,N  = 20
    ,C1 = 1.0
    ,C2 = 1.0
    ,ω  = 0.5
    ,options = Options(iterations=1000)
)

# Cota superior e inferior de individuos. alpha, beta, rho, sigma
lx = [0.0, 0.0, 0.0, 0.0 ]'
ux = [1.5, 1.5, 1.5, 1.5 ]'
lx_ux = vcat(lx,ux)

res = optimize( fitness, lx_ux, pso )


if _params[:wb]
    close(_params[:lg])
end
# EOF

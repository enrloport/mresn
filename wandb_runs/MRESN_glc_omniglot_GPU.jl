include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Logging
using Wandb


# Random.seed!(42)

# OMNIGLOT dataset

train = Omniglot(split=:train)
test  = Omniglot(split=:test)
cn = Omniglot().metadata["class_names"]

repit =500
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => true
    ,:wb_logger_name=> "MRESN_glc_omni_GPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1]
    ,:test_length   => size(test_y)[1]
    ,:train_f       => __do_train_MrESN_mnist!
    ,:test_f        => __do_test_MrESN_mnist!
    ,:B => 1.0
    ,:K => 0.01
)
const _B = _params[:B]
const _K = _params[:K]

#Generalised Logistic Curve
# A: Upper asymptote
# K: Lower asymptote
# C: Typically takes a value of 1. Otherwise, the upper asymptote is A + (K-A) / (C^(1/v))
# B: Growth rate
# v > 0: affects near which asymptote maximum growth occurs
# Q: is related to the value glc(0)
function glc(x; A=-_K, K=_K, C=1.0, B=_B, v=1.0, Q=1.0)
    return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/v) )
end

#for i in -10:10
#    v = i
#    println(v," ", glc(v))
#end

# b = 0.1
# max = 4
# x = [x for x in -1000:10:1000]
# y = [glc(i/50;B=b) for i in x ]
# ys = [ [glc(i/50;B=b/10) for i in x ] for b in 1:max]
# plot(x,ys; title="B influence in glc", labels= [b/10 for b in 1:max]' )


function do_batch(_params_esn, _params,sd)
    sz       = _params[:image_size]
    im_sz    = sz[1]*sz[2] 
    nodes    = _params_esn[:nodes] #im_sz
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    sgmds    = _params_esn[:sgmds]
    densities= _params_esn[:density]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]
    esns = [
        ESN( 
             R      = _params[:gpu] ? CuArray(new_R(nodes[i], density=densities[i], rho=rhos[i])) : new_R(nodes[i], density=densities[i], rho=rhos[i])
            ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], im_sz )) : rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], im_sz )
            ,R_scaling = r_scales[i]
            ,alpha  = alphas[i]
            ,rho    = rhos[i]
            ,sigma  = sigmas[i]
            ,sgmd   = sgmds[i]
            #,sgmd = glc
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
            ,train_function = _params[:train_f]
            ,test_function = _params[:test_f]
            )
        tm_train = @elapsed begin
            mrE.train_function(mrE,_params)
        end
	println("Maximum X: ", maximum(mrE.X))
        println("TRAIN FINISHED, ", tm_train)
        tm_test = @elapsed begin
            mrE.test_function(mrE,_params)
        end
        println("TEST FINISHED, ", tm_test)
    end
 
    to_log = Dict(
        "Total time" => tms
        ,"Train time"=> tm_train
        ,"Test time" => tm_test
       , "Error"     => mrE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
    end
    return mrE
end


#function transform_mnist(train_x, sz, trl)
#    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
#    trx = mapslices(x-> imresize(x, sz), trx[:,:,1:trl] ,dims=(1,2))
#    return trx
#end

function transform_mnist(train_x, sz, trl)
    trx = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, train_x)
    trx = mapslices(
        x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
    )
    return trx
end

#function transform_mnist(train_x, sz, trl)
#    trx = mapslices(
#            x-> imresize(x[ vec(mapslices(col -> any(col .!= 0), x, dims = 2)), vec(mapslices(col -> any(col .!= 0), x, dims = 1))], sz), train_x[:,:,1:trl] ,dims=(1,2)
#        )
#    trx[:] = map(x-> x > 0.3 ? 1.0 : x > 0.0 ? 0.5 : 0, trx)
#    return trx
#end



# r1      = 0
px      = 14 # rand([14,20,25,28])
sz      = (px,px)
train_x = transform_mnist(train_x, sz, _params[:train_length] )
test_x  = transform_mnist(test_x, sz, _params[:test_length])

for _ in 1:repit
    sd = rand(1:10000)
    Random.seed!(sd)
    _params[:num_esns] = 20 # rand([10,15,20,25])
    _params[:num_hadamard] = 0 # rand([1,2])
    max_d, min_d = 0.005, 0.2
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:alpha    => rand(Uniform(0.5,1.0),_params[:num_esns])
        #,:density  => rand(Uniform(0.01,0.7),_params[:num_esns])
        ,:density  => rand(Uniform(max_d, min_d),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => [1000 for _ in 1:_params[:num_esns] ] # rand([500, px*px ,1000],_params[:num_esns])
        #,:sgmds    => rand([glc, tanh, sigmoid],_params[:num_esns])
        ,:sgmds    => rand([ glc ],_params[:num_esns])
        #,:sgmds    => rand([ tanh ],_params[:num_esns])
    )
    _params[:initial_transient] = 1 #rand([1,2,3])
    _params[:image_size]   = sz
    _params[:train_data]   = train_x
    _params[:test_data]    = test_x
    _params[:train_labels] = train_y
    _params[:test_labels]  = test_y
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
        , "seed"               => sd
        , "sgmds"              => _params_esn[:sgmds]
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
	, "B" => _params[:B]
        , "K" => _params[:K]
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


# EOF

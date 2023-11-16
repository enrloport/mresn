include("../ESN.jl")
using MLDatasets
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
using Wandb


# Random.seed!(42)

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



repit = 1
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => true
    ,:wb_logger_name=> "MRESN_tanh_mnist_GPU_IT"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1] -55000
    ,:test_length   => size(test_y)[1]  -9000
    ,:train_f       => __do_train_MrESN_mnist!
    ,:test_f        => __do_test_MrESN_mnist!
)


# r1      = 0
px      = 14 # rand([14,20,25,28])
sz      = (px,px)
train_x = transform_mnist(train_x, sz, _params[:train_length] )
test_x  = transform_mnist(test_x, sz, _params[:test_length])



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
        ) for i in 1:_params[:num_esns]
    ]

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
	# println("Maximum X: ", maximum(mrE.X))
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
        ,"Seed" => sd
        ,"Alpha" => alphas[1]
        ,"Initial transient" => _params[:initial_transient]
        ,"Error"     => mrE.error
        ,"conf_mat" => Wandb.wandb.plot.confusion_matrix(
            y_true = test_y[1:_params[:test_length]], preds = [x[1] for x in mrE.Y], class_names = ["0","1","2","3","4","5","6","7","8","9"]
        )
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
    end
    return mrE
end




for _ in 1:repit
    sd = rand(1:10000)
    Random.seed!(sd)
    _params[:num_esns] = 5 # rand([10,15,20,25])
    #_params[:num_hadamard] = 0 # rand([1,2])
    min_d, max_d = 0.005, 0.2
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        # ,:alpha    => rand(Uniform(0.001,0.999),_params[:num_esns])
        ,:density  => rand(Uniform(min_d, max_d),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => [1000 for _ in 1:_params[:num_esns] ] # rand([500, px*px ,1000],_params[:num_esns])
        ,:sgmds    => [tanh for _ in 1:_params[:num_esns] ]
    )
    # _params[:initial_transient] = 1 #rand([1,2,3])
    _params[:image_size]   = sz
    _params[:train_data]   = train_x
    _params[:test_data]    = test_x
    _params[:train_labels] = train_y
    _params[:test_labels]  = test_y
    par = Dict(
        "Reservoirs" => _params[:num_esns]
        # ,"Hadamard reservoirs" => _params[:num_hadamard]
        , "Total nodes"        => sum(_params_esn[:nodes]) #+ sz[1]*sz[2] * _params[:num_hadamard]
        # , "Total nodes"       => _params[:num_esns] * _params_esn[:nodes] + sz[1]*sz[2] * _params[:num_hadamard]
        , "Train length"       => _params[:train_length]
        , "Test length"        => _params[:test_length]
        , "Resized"            => _params[:image_size][1]
        , "Nodes per reservoir"=> _params_esn[:nodes]
        # , "seed"               => sd
        , "sgmds"              => _params_esn[:sgmds]
        # , "alphas"             => _params_esn[:alpha]
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

    for a in [0.7, 0.9, 1.0]
        _params_esn[:alpha] = [ a for _ in 1:_params[:num_esns] ]

        for in_tr in 0:9
            _params[:initial_transient] = in_tr
            tm = @elapsed begin
                if _params[:wb]
                    _params[:lg] = wandb_logger(_params[:wb_logger_name])
                else
                    display(par)
                end
                r1 = do_batch(_params_esn,_params, sd)

                if _params[:wb]
                    close(_params[:lg])
                end
            end
        end

    end

end


# EOF

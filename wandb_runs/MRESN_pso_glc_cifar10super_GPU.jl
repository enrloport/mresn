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


# CIFAR-10 dataset
train = CIFAR10(split=:train)
test  = CIFAR10(split=:test)
cn = CIFAR10().metadata["class_names"]

train_x, train_y = train.features, train.targets
test_x,  test_y  = test.features, test.targets


# Classification based on superclass (animated or non-animated object)
train_y = map(x-> x in [0,1,8,9] ? 0 : 1 , train_y)
test_y = map(x-> x in [0,1,8,9] ? 0 : 1 , test_y)


px      = 32 # rand([14,20,25,28])
sz      = (px,px)



############################################################################ PARAMETERS


repit =1
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => false
    ,:wb_logger_name=> "MRESN_pso glc_cifar10super_GPU"
    ,:classes       => [0,1]
    ,:beta          => 1.0e-10
    ,:train_length  => size(train_y)[1]
    ,:test_length   => size(test_y)[1]
    ,:train_f       => __do_train_MrESN_cifar10!
    ,:test_f        => __do_test_MrESN_cifar10!
    ,:num_esns      => 25 # rand([10,15,20,25])
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
    ,:nodes    => [200 for _ in 1:_params[:num_esns] ]
)



############################################################################ LOG

par = Dict(
    "Reservoirs" => _params[:num_esns]
    , "Total nodes"        => sum(_params_esn[:nodes])
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
    , "preprocess"         => "no"
)



############################################################################ MRESN CREATION

esns = [
        ESN( 
             R      = _params[:gpu] ? CuArray(new_R(_params_esn[:nodes][i], density=_params_esn[:density][i], rho=_params_esn[:rho][i])) : new_R(_params_esn[:nodes][i], density=_params_esn[:density][i], rho=_params_esn[:rho][i])
            ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-_params_esn[:sigma][i],_params_esn[:sigma][i]), _params_esn[:nodes][i], sz[1]*sz[2]*3 )) : rand(Uniform(-_params_esn[:sigma][i],_params_esn[:sigma][i]), _params_esn[:nodes][i], sz[1]*sz[2]*3 )
            # R      = new_R(_params_esn[:nodes][i], density=_params_esn[:density][i], rho=_params_esn[:rho][i], gpu=_params[:gpu])
            # ,R_in   = new_R_in(_params_esn[:nodes][i], sz[1]*sz[2]; sigma=_params_esn[:sigma][i], density=1.0, gpu=_params[:gpu], channels=3)
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
        Wandb.log(_params[:lg], Dict("conf_mat" => Wandb.wandb.plot.confusion_matrix(
                # y_true = test_y[1:_params[:test_length]], preds = [x[1] for x in mrE.Y], class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
                y_true = test_y[1:_params[:test_length]], preds = [x[1] for x in mrE.Y], class_names = ["Non-Animated","Animated"]
            ))
        )
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

pso_dict = Dict(
    "N"  => 50
    ,"C1" => 1.0
    ,"C2" => 1.0
    ,"w"  => 0.5
    ,"max_iter" => 25
)

if _params[:wb]
    using Logging
    using Wandb
    _params[:lg] = wandb_logger(_params[:wb_logger_name])
    Wandb.log(_params[:lg], pso_dict )
else
    display(pso_dict)
    println(" ")
end

for _ in 1:repit
    if _params[:wb]
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    else
        display(par)
    end


    pso = PSO(;information=Metaheuristics.Information()
        ,N  = pso_dict["N"]
        ,C1 = pso_dict["C1"]
        ,C2 = pso_dict["C2"]
        ,ω  = pso_dict["w"]
        ,options = Options(iterations=pso_dict["max_iter"])
    )

    # Cota superior e inferior de individuos. k,b,v,q
    lx = [0.0001,0.0001,0.0001,0.0001,]'
    ux = [1.5, 1.5, 1.5, 1.5 ]'
    lx_ux = vcat(lx,ux)

    res = optimize( fitness, lx_ux, pso )


    if _params[:wb]
        close(_params[:lg])
    end
end

# EOF
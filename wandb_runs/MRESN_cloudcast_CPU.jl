####################  STRUCTS ####################

include("../ESN.jl")

using ImageEdgeDetection
using ImageEdgeDetection: Percentile
using ImageView
using Measures

using CUDA
CUDA.allowscalar(false)


dir     = "../data/cloudcast/"
file    = "TrainCloud.nc"
_data_o = ncread(dir*file, "__xarray_dataarray_variable__")

# CloudCast Preprocessing
_data = map(x-> x < 5 || x > 14 ? 0.0 : 1.0*(x-4) , _data_o )

_wb = false
gpu = true

trl = 10000
tel = 1000
it  = 500
px0 = (45,75)
rad = 3

_params_esn = Dict(
    :R_scaling => 1.0
    ,:alpha     => 0.6
    ,:beta      => 1.0e-10
    ,:rho       => 0.1
    ,:sigma     => 1.0
)

_params = Dict{Symbol, Any}(
     :pixel             => string(px0)
    ,:horizon           => 1
    ,:initial_transient => it
    ,:train_length      => trl
    ,:test_length       => tel
    ,:beta      => 1.0e-10
)

if _wb
    using Logging
    using Wandb
    lg = wandb_logger("grid_search__ESN_V2.x")
    Wandb.log(lg, _params)
    Wandb.log(lg, _params_esn)
end
    
_params[:train_function ]  = __do_train_pixel_forecast_subimage!
_params[:test_function  ]  = __do_test_pixel_forecast_subimage!
_params[:gpu]              = gpu


function do_batch(_data,rad,px0,_params,alpha,rho,dens,_wb)
    im_sz   = (2*rad + 1)
    nodes   = im_sz^2
    idx     = __subimage_index(rad, px0)
    trl,tel = _params[:train_length],  _params[:test_length]

    tr_d      = Array(_data[1:trl, idx[1]:idx[2], idx[3]:idx[4] ])
    te_d      = Array(_data[trl+1:trl+tel, idx[1]:idx[2], idx[3]:idx[4] ])
    te_l      = te_d[:,rad+1,rad+1]
    tr_l      = tr_d[:,rad+1,rad+1]

    _params[:train_data],_params[:train_labels],_params[:test_data],_params[:test_labels] = tr_d, tr_l, te_d, te_l
 
    Random.seed!(42) # SAME SEED to replicate experiments.
    R1    = new_R(nodes, density=dens, rho=rho)
    R_in1 = rand(Uniform(-1,1), nodes, 1 )
    R2    = new_R_img(im_sz,im_sz)
    R_in2 = rand(Uniform(-1,1), nodes, 1 )

    if _params[:gpu]
        R1       = CuArray(R1)
        R_in1    = CuArray(R_in1)
        R2       = CuArray(R2)
        R_in2    = CuArray(R_in2)
    end

    esn1 = ESN( R=R1, R_in=R_in1,
        R_scaling   = 1.0
        ,alpha      = alpha
        ,beta       = 1.0e-10
        ,sigma      = 1.0
    )

    copy_esn = deepcopy(esn1)

    ###############################################################################

    cny = Canny(spatial_scale=1, high=Percentile(90), low=Percentile(30))

    # te_d2 = mapslices( x -> detect_edges(x, cny) ,te_d, dims=(2,3))
    esn2 = ESN(
        R=R2
        , R_in=R_in2
    )
    esn2.F_in= (f,u) -> f( CuArray(u) .* CuArray(detect_edges(u, cny)) )
    # esn2.F_in= (f,u) -> esn2.R_in .* f(u)
    ###############################################################################

    tms = @elapsed begin
        res_mrE = MrESN(esns=[esn1] , beta=_params[:beta])
        __do_train_MrESN!(res_mrE,_params)
        __do_test_MrESN!(res_mrE,_params)
    end
    tms2 = @elapsed begin
        res_esn = run_esn_struct(
            ;esn = copy_esn
            ,_params...
        )
    end

    print("\n", tms, ", ", tms2 , "\n")
    if _wb
        Wandb.log(lg, Dict( "Time"=>tms, "Error"=>err, "density"=>dens, "alpha"=> alpha, "rho"=>rho, "Radius"=>rad, "Train length" => trl, "Nodes"=>nodes ) )
    end

    return res_mrE, res_esn
end


include("../ESN.jl")
res_mre, res_esn = do_batch(_data,rad,px0,_params,0.2,1.0,0.6,_wb)


te_l = _data[trl+1:trl+tel, px0...]

size(res_mre.R_out)
size(res_esn.R_out)

res1 = Array(res_mre.Y')
res2 = Array(res_esn.Y')

err1 = mse( res1 , te_l )
err2 = mse( res2 , te_l )



plts = plot_MrESN(res_mre, te_l, [false,true,false,true]; size=(1800,600), layout=[1;1])

# savefig( plts, "hola.svg")    
# Plots.plot(plts...)








idx     = __subimage_index(40, px0)
te_d      = Array(_data[trl+1:trl+tel, idx[1]:idx[2], idx[3]:idx[4] ])


cny = Canny(spatial_scale=1, high=Percentile(90), low=Percentile(30))

te_d2 = mapslices( x -> detect_edges(x, cny) ,te_d, dims=(2,3))

te_d3 = te_d.*te_d2

ImageView.imshow(te_d, axes=(2,3))
ImageView.imshow(te_d2, axes=(2,3))
ImageView.imshow(te_d3, axes=(2,3))


plot( [res1,res2,te_l]
    , size=(2200,350)
    , margin=12mm
    , label=["ESN" "MrESN" "Target"]
    , xlabel="steps"
    , ylabel="value"
    )
    
plot( [der1,der2,te_l]
    , size=(2200,350)
    , margin=12mm
    , label=["ESN" "MrESN" "Target"]
    , xlabel="steps"
    , ylabel="value"
    )

if _wb
    close(lg)
end



####################  STRUCTS ####################
using Logging
using Wandb

include("ESN.jl")

using CUDA
using Measures

CUDA.allowscalar(false)

gpu = true
trl = 50000
tel = 1000
it  = 1000

full_img_height = 128
full_img_widtht = 128

radius = 3
im_sz = (2*radius + 1)^2
nodes = im_sz
R_sz  = nodes*nodes

dir = "../data/cloudcast/splitted_radius_3/"
file = "cloudcast_EQ_radius-" * string(radius) * "_("
ext  = ").csv"


_params = Dict{Symbol, Any}(
    :nodes             => nodes
    ,:density           => 0.2
    ,:initial_transient => it
    ,:horizon           => 1
    ,:alpha             => 0.6
    ,:beta              => 1.0e-10
    ,:train_length      => trl
    ,:test_length       => tel
    ,:initial_transient => it
    ,:train_labels      => []
    ,:test_labels       => []
    ,:train_function    => __do_train_pixel_forecast_subimage!
    ,:test_function     => __do_test_pixel_forecast_subimage!
    ,:radius            => radius
)


lg = wandb_logger("cloudcast_subimage")
Wandb.log(lg, _params)

# SAME RESERVOIR BUT DIFFERENT INPUT LAYERS
Random.seed!(42)
R    = new_R(_params[:nodes], density=_params[:density], rho=0.1)


file_name = dir*file*"4,4"*ext
_data = CSV.read(file_name, DataFrame; header=false)
tex0 = reshape(Array(_data[trl+1:trl+tel,:]), :, (2*radius +1), (2*radius +1) )
trx0 = reshape(Array(_data[1:trl,:]), :, (2*radius +1), (2*radius +1))
target = tex0[:,1,1]

for row in 1+radius:full_img_height-radius
    for col in 1+radius:full_img_widtht-radius

        pxl  = string(row) * "," * string(col)
        file_name = dir*file*pxl*ext

        _data[:,:] = CSV.read(file_name, DataFrame; header=false)
        tex0[:] = reshape(Array(_data[trl+1:trl+tel,:]), :, (2*radius +1), (2*radius +1) )
        trx0[:] = reshape(Array(_data[1:trl,:]), :, (2*radius +1), (2*radius +1))
        target[:] = tex0[:,row,col]

        # New R_in
        R_in = rand(Uniform(-1,1), _params[:nodes], 1 ) 

        if gpu
            R    = CuArray(R)
            R_in = CuArray(R_in)
        end

        # lg = wandb_logger("cloudcast_subimage")
        # Wandb.log(lg, _params)

        esn = ESN(
            R         = R
            ,R_in      = R_in
            ,R_scaling = 1.0
            ,alpha     = 0.6
            ,beta      = 1.0e-10
            ,rho       = 0.1
            ,sigma     = 1.0
        )

        t1 = now().instant.periods.value/1000
        @time res = run_esn_struct(
            ; esn
            ,_params...
            ,pixel = (row,col)
            ,train_data = trx0
            ,test_data = tex0
            , gpu = gpu
        )
        t2 = now().instant.periods.value/1000
        tms = t2-t1
        err = sum(abs.( (res.Y').*(res.Y') .- target.*target) / length(target) )

        Wandb.log(lg, Dict( "row"=>row, "column"=>col, "time"=>tms, "MSE"=>err ) )
        

        wandb_log_artif(lg, "R_out_pixel__" * string(row) * "_" * string(col) , res.R_out, type = "dataset")
        wandb_log_artif(lg, "predictions_pixel__" * string(row) * "_" * string(col) , res.Y, type = "dataset")

    end
end
close(lg)


# using Measures
# plot(
#     [res.Y',target] , size = (800, 300) , margin=5mm
#     , title="Original dataset. Int values."
#     , labels=["prediction" "target"]
#     , xlabel="Steps"
#     , ylabel="Value"
# )






#####################



# This function fills the matrix X using CuArrays if gpu param is true.
# Input:
#   - args: Dictionary with main parameters.
#   - x: state Vector
#   - u: input
#   - X: the matrix to be filled.
function __fill_X!(args::Dict, x::Union{Array, CuArray}, u::Union{Array, CuArray}, X::Union{Array, CuArray} )
    if args[:gpu]
        for t = 1:args[:initial_transient]
            u[:] = CuArray(reshape(args[:train_data][t,:,:], :, 1))
            __update_x!(args[:esn].R_in,args[:esn].R,x,u,args[:esn].alpha)
        end
        for t = args[:initial_transient]+1:size(args[:train_data],1) - args[:horizon]
            u[:] = CuArray(reshape(args[:train_data][t,:,:], :, 1))
            __update_x!(args[:esn].R_in,args[:esn].R,x,u,args[:esn].alpha)
            X[:,t-args[:initial_transient]] = args[:new_column](u,x)
        end
    else
        for t = 1:args[:initial_transient]
            u[:] = reshape(args[:train_data][t,:,:], :, 1)
            __update_x!(args[:esn].R_in,args[:esn].R,x,u,args[:esn].alpha)
        end
        for t = args[:initial_transient]+1:size(args[:train_data],1) - args[:horizon]
            u[:] = reshape(args[:train_data][t,:,:], :, 1)
            __update_x!(args[:esn].R_in,args[:esn].R,x,u,args[:esn].alpha)
            X[:,t-args[:initial_transient]] = args[:new_column](u,x)
        end
    end
end
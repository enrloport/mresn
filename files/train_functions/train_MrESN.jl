function sigmoid(x)
    return 1 / (1 + exp(-x))
end

function update(esn,u, f)
    # println(typeof(esn.R_in))
    # println(typeof(u))
    # println(typeof(esn.R))
    # println(typeof(esn.x))
    esn.x[:] = (1-esn.alpha).*esn.x .+ esn.alpha.*esn.sgmd.( esn.F_in(f,u) .+ esn.R*esn.x)
end

# This function fills the matrix X using CuArrays if gpu param is true.
# Input:
#   - mrE: Multi reservoir struct
#   - args: Dictionary with main parameters.
#   - u: input
function __fill_X_MrESN!(mrE, args::Dict )

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t = 1:args[:initial_transient]
        for i in 1:length(mrE.esns)
            update(mrE.esns[i], args[:train_data][t,:,:], f )
        end
    end
    for t = args[:initial_transient]+1:size(args[:train_data],1) - args[:horizon]
        for i in 1:length(mrE.esns)
            update(mrE.esns[i], args[:train_data][t,:,:], f )
        end
        mrE.X[:,t-args[:initial_transient]] = vcat(f(args[:train_data][t,:,:]),[es.x for es in mrE.esns]...)
    end
end


function __make_Rout_MrESN!(mrE,args, train_labels)
    X    = mrE.X
    min  = args[:initial_transient]+1+args[:horizon]
    if gpu
        train_labels = CuArray(train_labels)
        mrE.R_out = CuArray( transpose( (X*transpose(X) + mrE.beta*I)\(X*train_labels[min:end]) ) )
    else
        mrE.R_out = Matrix( transpose( (X*transpose(X) + mrE.beta*I)\(X*train_labels[min:end]) ) )
    end
end


function __do_train_MrESN!(mrE, args)
    num   = size(args[:train_data],1)-args[:initial_transient]-args[:horizon]
    # u     = reshape(args[:train_data][1,:,:], :, 1)
    mrE.X = zeros( mrE.esns[1].R_size*(length(mrE.esns)+1) , num)

    for i in 1:length(mrE.esns)
        mrE.esns[i].x = zeros( mrE.esns[i].R_size, 1)
    end

    if args[:gpu]
        mrE.X = CuArray(mrE.X)
        for i in 1:length(mrE.esns)
            mrE.esns[i].x = CuArray(mrE.esns[i].x)
        end
    end

    __fill_X_MrESN!(mrE,args)
    __make_Rout_MrESN!(mrE,args, args[:train_labels] )
end
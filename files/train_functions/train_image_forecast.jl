function __make_Routs!(args)
    X    = args[:result][:X]
    min  = args[:initial_transient]+1+args[:horizon]
    n_px = size(args[:train_data],2)

    if args[:gpu]
        data = CuArray(args[:train_data][:,1])
        args[:result][:R_outs] = CuArray(zeros( n_px, size(X,1) ))
        for pix in 1:n_px
            data[:] = CuArray(args[:train_data][:,pix])
            args[:result][:R_outs][pix:pix,:] = Matrix( transpose( (X*transpose(X) + args[:beta]*I)\(X*data[min:end]) ) )
        end
    else
        data = args[:train_data][:,1]
        args[:result][:R_outs] = zeros( n_px, size(X,1) )
        for pix in 1:n_px
            data[:] = args[:train_data][:,pix]
            args[:result][:R_outs][pix:pix,:] = Matrix( transpose( (X*transpose(X) + args[:beta]*I)\(X*data[min:end]) ) )
        end
    end
end


# Similar to pixel forecast function, but in this case we apply forecasting to all pixels of the image
function __do_train_image_forecast!( args::Dict )
    num  = size(args[:train_data],1)-args[:initial_transient]-args[:horizon]
    if args[:gpu]
        args[:result][:X] = CuArray(zeros(args[:R_size]*2+args[:zero_degree], num))
        args[:result][:x] = CuArray(zeros(args[:R_size], 1))
        u                 = CuArray(reshape(args[:train_data][1,:,:], :, 1))
    else
        args[:result][:X] = zeros(args[:R_size]*2+args[:zero_degree], num)
        args[:result][:x] = zeros(args[:R_size], 1)
        u                 = reshape(args[:train_data][1,:,:], :, 1)
    end

    __fill_X!(args,args[:result][:x], u, args[:result][:X])
    __make_Routs!(args)

    return args
end
#=
    If a trained ESN is given as input in the main function, trained parameters are propagateded to test phase.
=#
function __no_train(args::Dict)
    return args
end


function __no_train_update_x!(args::Dict)
    num = size(args[:train_data],1)-args[:initial_transient]-args[:horizon]
    
    u = reshape(args[:train_data][1,:,:], :, 1)
    args[:esn].X = zeros( args[:esn].R_size*2+args[:zero_degree], num)
    args[:esn].x = zeros( args[:esn].R_size, 1)

    if args[:gpu]
        u, args[:esn].X , args[:esn].x = CuArray(u), CuArray(args[:esn].X), CuArray(args[:esn].x) 
        for t = 1:size(args[:train_data],1) - args[:horizon]
            u[:] = CuArray(reshape(args[:train_data][t,:,:], :, 1))
            __update_x!(args[:esn].R_in,args[:esn].R,args[:esn].x,u,args[:esn].alpha)
        end
    else
        for t = 1:size(args[:train_data],1) - args[:horizon]
            u[:] = Array(reshape(args[:train_data][t,:,:], :, 1))
            __update_x!(args[:esn].R_in,args[:esn].R,args[:esn].x,u,args[:esn].alpha)
        end
    end
    return args
end

#=
    This function is used by default for training. When R_fdb is not gived as input in main function (run_esn), a Matrix of zeros 
    is used so, no feedback is applied by default.
    Alpha and Beta params are used in this function.
=#
function __do_train!( args::Dict )
    data             = args[:train_data]
    W                = args[:R]
    initial_transient= args[:initial_transient]
    train_length     = args[:train_length]
    alpha            = args[:alpha]
    beta             = args[:beta]
    in_size          = args[:in_size]
    out_size         = args[:out_size]
    R_size           = args[:R_size]
    Win              = args[:R_in]
    Wf               = args[:R_fdb]
    Y_train          = args[:train_labels] #[initial_transient+1:train_length, :] #data[initial_transient+2:train_length+1]
    X                = args[:X]
    x                = args[:x]

    u = data[1, 1:end]
    x = (1-alpha).*x .+ alpha.*tanh.( Win*u .+ W*x)      # First step. No feedback    

    for t = 2:train_length
        u = data[t, 1:end]
        x = (1-alpha).*x .+ alpha.*tanh.( Win*u .+ W*x .+ Wf*data[t-1, out_size] )

        if t > initial_transient
            X[:,t-initial_transient] = [u;x]
        end
    end

    args[:result][:x],args[:result][:X] = x,X
    if all( args[:result][:R_out] .== 0.0)
        args[:result][:R_out] = Matrix(transpose((X*transpose(X) + beta*I) \ (X*Y_train)))
    end

    return args
end


function __do_batch_train!( args::Dict )
    data             = args[:train_data]
    Ws               = args[:Rs]

    for r in Ws
        W                = r[:R]
        Win              = r[:R_in]
        Wf               = r[:R_fdb] 
        R_size           = r[:R_size]
        initial_transient= r[:initial_transient]
        train_length     = r[:train_length]
        in_size          = r[:in_size]
        out_size         = r[:out_size]
        alpha            = r[:alpha]
        beta             = r[:beta]
        Y_train          = data[initial_transient+2:train_length+1]
        X                = zeros(in_size+R_size , train_length-initial_transient)
        x                = zeros(R_size, 1)

        x                = (1-alpha).*x .+ alpha.*tanh.( Win*data[1, 1:end] .+ W*x)      # First step.
        for t = 2:train_length
            u = data[t, 1:end]
            x = (1-alpha).*x .+ alpha.*tanh.( Win*u .+ W*x .+ Wf*data[t-1, 1:out_size] )

            if t > initial_transient
                X[:,t-initial_transient] = [u;x]
            end
        end
        r[:X], r[:x] = X,x
        if all( r[:R_out] .== 0.0)
            r[:R_out] = transpose((X*transpose(X) + beta*I) \ (X*Y_train))
        end
    end

    return args
end


#=
    Default training for esn_img_classifier function.
    The input image is resized to a vector and its product with R_in is gived to the reservoir.
    The states are set to zero at the begining of each step and only one step is applied as initial transient.
    Feedback is not used in this function.
=#
function __do_train_img!( args::Dict )
    data             = args[:train_data]
    labels           = args[:train_labels]
    initial_tr       = args[:initial_transient]
    classes          = args[:classes]
    W                = args[:R]
    sigma            = args[:sigma]
    alpha            = args[:alpha]
    beta             = args[:beta]
    R_size           = args[:R_size]
    Win              = args[:R_in]
    Wf               = args[:R_fdb]

    train_length = size(data,3)
    X            = zeros(R_size*2, train_length)    
    x            = zeros(R_size, 1)    
    classes_Yt   = Dict( c => Int[] for c in classes )  # New dataset for each class

    for t = 1:train_length
        u = data[:,:,t]
        u = reshape(u, :, 1)

        x  = zeros(R_size, 1) # Reset states before see new image.

        for _ in 1:initial_tr
            x  = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x )
        end
        x      = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x )

        X[:,t] = [u;x]
        
        lt = labels[t]
        for c in classes
            y = lt == c ? 1 : 0
            push!(classes_Yt[c], y)
        end
    end
    # classes_Rout is a dict with class numbers as keys and the computed R_out for each class as values.
    classes_Rout = Dict( c => Matrix(transpose((X*transpose(X) + beta*I) \ (X*classes_Yt[c]))) for c in classes )

    args[:result][:x] = x
    args[:result][:X] = X    
    args[:result][:classes_Yt]   = classes_Yt
    args[:result][:classes_Rout] = classes_Rout

    return args
end


#=
    Same as __do_train_img but this function applies feedback.
=#
function __do_train_img_fb!( args::Dict )
    data         = args[:train_data]
    labels       = args[:train_labels]
    initial_tr   = args[:initial_transient]
    classes      = args[:classes]
    W            = args[:R]
    sigma        = args[:sigma]
    alpha        = args[:alpha]
    beta         = args[:beta]
    R_size       = args[:R_size]
    Win          = args[:R_in]
    Wf           = args[:R_fdb]

    train_length = size(data,3)
    X            = zeros(R_size*2, train_length)    
    x            = zeros(R_size, 1)    
    classes_Yt   = Dict( c => Int[] for c in classes )  # New dataset for each class

    for t = 1:train_length
        lt = labels[t]
        u  = reshape(data[:,:,t], :, 1)
        x  = ones(R_size, 1) # Reset states before see new image.

        for i in 1:initial_tr
            blur_u = blur_img(data[:,:,t], initial_tr+1 - i) # Blured image is shown to the reservoir as feedback
            blur_u = reshape(blur_u, :, 1)
            x      = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x .+ Wf.*blur_u)
        end
        x      = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x )
        X[:,t] = [u;x]

        for c in classes
            y = lt == c ? 1 : 0
            push!(classes_Yt[c], y)
        end
    end

    # classes_Rout is a dict with class numbers as keys and the computed R_out for each class as values.
    classes_Rout = Dict( c => Matrix(transpose((X*transpose(X) + beta*I) \ (X*classes_Yt[c]))) for c in classes )

    args[:result][:x] = x
    args[:result][:X] = X    
    args[:result][:classes_Yt]   = classes_Yt
    args[:result][:classes_Rout] = classes_Rout

    return args
end


#=
    In every step of initial transient a sharper image (same input) is shown to the reservoir.
=#
function __do_train_img_blur!( args::Dict )
    data         = args[:train_data]
    data_blur    = args[:train_data_blur]
    labels       = args[:train_labels]
    initial_tr   = args[:initial_transient]
    classes      = args[:classes]
    W            = args[:R]
    sigma        = args[:sigma]
    alpha        = args[:alpha]
    beta         = args[:beta]
    R_size       = args[:R_size]
    Win          = args[:R_in]

    train_length = size(data,3)
    X            = zeros(R_size*2, train_length)    
    x            = zeros(R_size, 1)    
    classes_Yt   = Dict( c => Int[] for c in classes )  # New dataset for each class

    @assert(size(Win) == size(data[:,:,1]), "Dimension mismatch: R_in " * string(size(Win)) * ", data[:,:,1] " * string(size(data[:,:,1])) )
    @assert(size(W,1) == size(x,1)        , "Dimension mismatch: R " * string(size(W)) * ", x " * string(size(x)) )
    for t = 1:train_length
        lt = labels[t]
        # u  = reshape(data[:,:,t], :, 1)
        u  = data[:,:,t]
        x  = ones(R_size, 1) # Reset states before see new image.

        for i in 1:initial_tr
            # blur_u0 = blur_img(data[:,:,t], initial_tr+1 - i) # Blured image is shown to the reservoir
            # blur_u0 = reshape(blur_u0, :, 1)
            blur_u = data_blur[initial_tr+1-i][:,:,t]
            x      = (1-alpha).*x .+ alpha.*tanh.( Win.*blur_u .+ W*x)
        end
        x      = (1-alpha).*x .+ alpha.*tanh.( Win.*u .+ W*x )
        X[:,t] = [u;x]

        for c in classes
            y = lt == c ? 1 : 0
            push!(classes_Yt[c], y)
        end
    end

    # classes_Rout is a dict with class numbers as keys and the computed R_out for each class as values.
    classes_Rout = Dict( c => Matrix(transpose((X*transpose(X) + beta*I) \ (X*classes_Yt[c]))) for c in classes )

    args[:result][:x] = x
    args[:result][:X] = X    
    args[:result][:classes_Yt]   = classes_Yt
    args[:result][:classes_Rout] = classes_Rout

    return args
end


function __do_train_img_fine!( args::Dict )
    X          = zeros(args[:R_size]*2, length(args[:bad_classified]) )
    x          = zeros(args[:R_size], 1)
    classes_Yt = Dict( c => Int[] for c in args[:classes] )  # New dataset for each class

    i = 0
    for t in args[:bad_classified] # Only previously bad classify examples are shown to the network.
        i += 1
        u = args[:train_data][:,:,t]
        u = reshape(u, :, 1)
        x = __compute_state_img(args[:R_in], args[:R], u, zeros(args[:R_size],1 ), args[:alpha], args[:initial_transient] )

        X[:,i] = [u;x]
        lt = args[:train_labels][t]
        for c in args[:classes]
            y = lt == c ? 1 : 0
            push!(classes_Yt[c], y)
        end
    end
    # classes_Rout is a dict with class numbers as keys and the computed R_out for each class as values.
    classes_Rout = Dict( 
        c => Matrix(transpose((X*transpose(X) + args[:beta] *I) \ (X*classes_Yt[c]))) for c in args[:classes] 
        )
    esn = args[:result]
    esn[:x], esn[:X],esn[:classes_Yt],esn[:classes_Rout] = x,X,classes_Yt,classes_Rout
    return args
end


function __do_train_img_child!( args::Dict )
    classes      = args[:classes]
    x            = zeros(length(classes)*length(args[:parents]), 1)
    X            = zeros(length(classes)*length(args[:parents])*2, size(args[:train_data],2))
    classes_Yt   = Dict( c => Int[] for c in classes )  # New dataset for each class

    for t = 1:size(args[:train_data],2)
        u = args[:train_data][:,t]
        x = __compute_state_img(args[:R_in], args[:R], u, zeros(length(u),1 ), args[:alpha], args[:initial_transient] )
        X[:,t] = [u;x]

        lt = args[:train_labels][t]
        for c in classes
            y = lt == c ? 1 : 0
            push!(classes_Yt[c], y)
        end
    end
    classes_Rout = Dict( c => Matrix(transpose((X*transpose(X) + args[:beta]*I) \ (X*classes_Yt[c]))) for c in classes )

    esn = args[:result]
    esn[:x], esn[:X], esn[:classes_Yt], esn[:classes_Rout] = x, X, classes_Yt, classes_Rout
    return args
end

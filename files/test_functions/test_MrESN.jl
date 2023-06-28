
# Function to test an already trained MrESN struct
function __do_test_MrESN!(mrE, args::Dict)
    test_length = args[:test_length]
    mrE.Y       = zeros( 1, test_length) # Predictions matrix

    if gpu
        mrE.Y = CuArray(mrE.Y)
        f = (u) -> CuArray(reshape(u, :, 1))
    else
        f = (u) -> reshape(u, :, 1)
    end

    for t in 1:args[:horizon]  # Last examples of training
        for i in 1:length(mrE.esns)
            update(mrE.esns[i], args[:train_data][end-(args[:horizon]-t),:,:], f )
        end
        mrE.Y[:,t] = mrE.R_out*vcat( f(args[:train_data][end-(args[:horizon]-t),:,:]) , [es.x for es in mrE.esns]...)
    end
    for t = 1:test_length-args[:horizon]
        for i in 1:length(mrE.esns)
            update(mrE.esns[i], args[:test_data][t,:,:], f )
        end
        mrE.Y[:,t] = mrE.R_out*vcat(f(args[:test_data][t,:,:]), [es.x for es in mrE.esns]...)
    end
end
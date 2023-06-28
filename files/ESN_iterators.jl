function sparsity_iterator!(
    ; train_data::Data
    , res::Dict
    , density::Float64
    , selector::Function=rand_selection!
    , final_train::Bool=false
    , kwargs...
    )

    m, n       = size(res[:R])
    tam_R      = m*n
    to_rmv     = tam_R - floor(Int, tam_R*density)
    aux        = deepcopy(res)
    candidates = [(i,j) for i in 1:m for j in 1:n]
    for i in 1:to_rmv
        aux, candidates = selector(aux, candidates; data=train_data, kwargs...)
    end

    if final_train
        aux = run_esn(
            initial_transient = aux[:initial_transient]
            , train_length    = aux[:train_length]
            , test_length     = aux[:test_length]
            , train_data      = train_data
            , alpha           = aux[:alpha]
            , beta            = aux[:beta]
            , rho             = aux[:rho]
            , sigma           = aux[:sigma]
            , R_scaling       = aux[:R_scaling]
            , R               = aux[:R]
            , R_size          = aux[:R_size]
            , R_in            = aux[:R_in]
            , R_fdb           = aux[:R_fdb]
            , kwargs...
        )
    end

    return aux
end


function rand_selection!(res::Dict, candidates::Vector; kwargs...)
    index      = rand(1:size(candidates,1))
    i, j       = __pop!(candidates, index)
    
    res[:R][i,j] = 0.0 # If we change some value in the reservoir, we must repeat trainning to obtain x,X,Y and error
    res[:X]      = []
    res[:Y]      = []
    res[:x]      = []
    res[:error]  = Inf16

    return res, candidates
end


function best_of_n!(result, candidates; n, data, auto_adjust_spectral_radius=false, kwargs...)
    sz    = size(candidates,1)
    batch = sz > n ? n : sz
    indxs = rand(1:sz , batch ) # Take n random indexes from candidates
    
    R_candidates = [ break_path!( copy(result[:R]) , candidates[i] ) for i in indxs  ]
       
    if auto_adjust_spectral_radius
        R_candidates = [set_spectral_radius!(x,result[:rho]) for x in R_candidates]
    end

    trained_candidates = run_batch_esn!(
        train_data         = data
        ,Rs                = R_candidates
        ,initial_transient = result[:initial_transient]
        ,train_length      = result[:train_length]
        ,test_length       = result[:test_length]
        ,R_scaling         = result[:R_scaling]
        ,alpha             = result[:alpha]
        ,beta              = result[:beta]
        ,sigma             = result[:sigma]
        ,rho               = result[:rho]
        ,R_fdb             = result[:R_fdb]
        ,R_in              = result[:R_in]
    )
    best = (trained_candidates[1], indxs[1])

    for i in 2:n
        r, c = trained_candidates[i], indxs[i]

        if r[:error] < best[1][:error]
            best = (r, c)
        end       
    end
    deleteat!(candidates, best[2])

    return best[1], candidates
end
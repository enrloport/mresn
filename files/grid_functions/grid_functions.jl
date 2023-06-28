function __update_best(out_file, train_length, test_length, alpha, beta, initial_transient, rho, sigma, R_scaling, error; log=true, verbose=true)
    s = ""
    s = s * "\n\n"* __time_now() *" - Current Parameters:"
    s = s * "\n\t,:initial_transient => "* string(initial_transient)
    s = s * "\n\t,:train_length => " 	 * string(train_length)
    s = s * "\n\t,:test_length => " 	 * string(test_length)
    s = s * "\n\t,:R_scaling => " 		 * string(R_scaling)
    s = s * "\n\t,:alpha => " 		     * string(alpha)
    s = s * "\n\t,:beta => " 			 * string(beta)
    s = s * "\n\t,:sigma => " 	         * string(sigma)
    s = s * "\n\t,:rho => " 			 * string(rho)
    s = s * "\n--> Error: " 	         * string(error)
    s = s * "\n"
    if log
        result_file  = open(out_file, "a")
        write(result_file, s)
        close(result_file)
    end
    if verbose
        print(s)
    end
    return s
end


function grid_search_esn(
    ; R                      ::Mtx             = zeros(1,1)
    , R_in                   ::Mtx             = zeros(1,1)
    , R_scaling_list         ::Array{Float64}  = [1.0]
    , alpha_list             ::Array{Float64}  = [0.3, 0.7]
    , beta_list              ::Array{Float64}  = [1.0e-3, 1.0e-9]
    , rho_list               ::Array{Float64}  = [0.7, 1, 1.3]
    , sigma_list             ::Array{Float64}  = [1.0]
    , main_function          ::Function        = run_esn
    , wandb_logger                             = ""
    , kwargs...
    )

    t0 = __time_now()
    print("\n\n\n Starting Grid Search: ", t0)

    best_R       = Dict()
    current_best = Inf16

    for R_scale in R_scaling_list
        print("\n Main loop. Current scale: ", R_scale )
        for sigma in sigma_list
            for beta in beta_list
                for alpha in alpha_list
                    for rho in rho_list
                        current_R = main_function(
                            ; R_scaling         = R_scale
                            , alpha             = alpha
                            , beta              = beta
                            , rho               = rho
                            , sigma             = sigma
                            , R                 = deepcopy(R)
                            , R_in              = R_in
                            , kwargs...
                        )
                        error = current_R[:error]
                        if error < current_best
                            current_best = error
                            best_R       = deepcopy(current_R)
                        end

                        if wandb_logger != ""
                            Wandb.log(wandb_logger , Dict("error" => error ,"alpha" => alpha ,"beta" => beta ,"rho" => rho ) )
                        end
                    end
                end
            end
        end
    end
    t1 = __time_now()
    print("\n\n\n Grid Search has finished. Time: ", t1)
    return best_R
end

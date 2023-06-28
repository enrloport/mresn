

function new_R( R_size::Int=50; R_scaling::Float64=1.0, rho::Float64=1.0, density=1.0, distribution=Uniform)
    if density != 1.0
        W = sprand(R_size, R_size, density, x-> rand(distribution(-R_scaling, R_scaling), x) )
        W = Array(W)
    else
        W = rand( distribution( -R_scaling, R_scaling ) , R_size, R_size )
    end
    set_spectral_radius!( W , rho)

    return W
end
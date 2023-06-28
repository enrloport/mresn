
function __plot_ESN(dict; show_n=[], show_signal=true, show_prediction=true, show_nodes=true, show_R_out=true, title="" )
    try
        data    = dict[:Y_target][:,1]
        R_out   = transpose(dict[:R_out])
        X       = transpose(dict[:X])
        Y       = transpose(dict[:Y])
        max_n   = size(X,2) < 5 ? size(X,2) : 5
        sz      = size(R_out,1)
    
        to_show = []
    
        if show_signal
            plt1    = Plots.plot( data, title="Signal", reuse=false, leg=false)
            to_show = [plt1]
        end
        if show_prediction
            plt2    = Plots.plot( [data,Y], title="Prediction", reuse=false, leg=false)
            to_show = cat(to_show, plt2, dims=1)
        end
        if show_nodes
            plt3    = Plots.plot( X[:,1:max_n], title="Nodes", reuse=false, leg=false)
            to_show = cat(to_show, plt3, dims=1)
        end
        if show_R_out
            plt4    = Plots.bar( 1:sz, R_out, title="R_out", reuse=false, leg=false)
            to_show = cat(to_show, plt4, dims=1)
        end
    
        plts =  cat(  to_show 
                    , [ Plots.plot( X[ :,[n]], title="N"*string(n), reuse=false, leg = false ) for n in show_n ]
                    , dims=1
                )
        return plts
    catch e
        print("__plot_ESN exception.  Is the network trained and tested?")
        return
    end
end

function plot_ESNs(ESNs...; show_n=[], show_signal=true, show_prediction=true, show_nodes=true, show_R_out=true, kwargs...)
    rows  = size(show_n,1) + sum([show_signal && 1, show_prediction && 1, show_nodes && 1, show_R_out && 1])
    plts  = []
    order = []
    errors= []
    i     = 0
    for R in ESNs
        i  += 1
        plt = __plot_ESN(R, show_n=show_n, show_signal=show_signal, show_prediction=show_prediction, show_nodes=show_nodes, show_R_out=show_R_out)
        push!(plts,plt)
        push!(errors, ("R"*string(i), R[:error]) ) 
    end
    for r in 1:rows, p in plts
        push!(order, p[r])
    end

    len1 = rows
    len2 = size(ESNs,1)
    res  = Plots.plot(order... , layout=(len1,len2) ; kwargs...)

    display(errors)
    return res
end



function plot_MrESN(mrE, data, sh=[true,true,true,true]; show_n=[], kwargs... )
    
    show_signal, show_prediction, show_nodes, show_R_out = sh[1], sh[2], sh[3], sh[4]
    # try
        data    = Array(data)
        R_out   = Array(mrE.R_out')
        X       = Array(mrE.X')
        Y       = Array(mrE.Y')
        max_n   = size(X,2) < 5 ? size(X,2) : 5
        sz      = size(R_out,1)
    
        to_show = []
    
        if show_signal
            plt1    = Plots.plot( data, title="Signal", reuse=false, leg=false)
            to_show = [plt1]
            xlabel  = "step"
            ylabel  = "value"
        end
        if show_prediction
            plt2    = Plots.plot( [data, Y], title="Prediction", reuse=false, leg=false)
            to_show = cat(to_show, plt2, dims=1)
            xlabel  = "step"
            ylabel  = "value"
        end
        if show_nodes
            plt3    = Plots.plot( X[:,1:max_n], title="Nodes", reuse=false, leg=false)
            to_show = cat(to_show, plt3, dims=1)
            xlabel  = "step"
            ylabel  = "value"
        end
        if show_R_out
            plt4    = Plots.bar( 1:sz, R_out, title="R_out", reuse=false, leg=false)
            to_show = cat(to_show, plt4, dims=1)
            xlabel  = "node"
            ylabel  = "value"
        end
    
        plts =  cat(  to_show 
                    , [ Plots.plot( X[ :,[n]], title="N"*string(n), reuse=false, leg = false ) for n in show_n ]
                    , dims=1
                )
        return Plots.plot(plts...;kwargs...)
    # catch e
    #     print("plot_MrESN exception.  Is the network trained and tested?")
    #     return
    # end
end


########################################################

include("ESN.jl")

using Plots
using PlotlyJS

#########################################



#################    BAR PLOTS

# cpu1 = [
# 46.377200,
#  43.584966,
#  44.409999,
#  44.066787,
#  44.393150,
#  44.547119,
#  44.375864,
#  44.382110,
#  44.449798,
#  44.420971
# ]

# cpu2 = [
#     2203.737918,
#     2215.152544,
# 2254.368758,
# 2227.781206,
# 2228.847995,
# 2174.376256,
# 2227.499832,
# 2183.431484,
# 2230.864879,
# 2174.945465,
# ]

# mean(cpu1)
# mean(cpu2)

# gpu1 = [
#     9.144303,
#     8.907645,
#     8.929718,
#     9.157349,
#     8.887713,
#     9.230467,
#     8.939521,
#     8.969454,
#     9.209229
# ]

# gpu2 = [
#     142.901627,
#     116.338565,
# 116.945886,
# 116.772491,
# 116.646275,
# 116.634803,
# 117.508326,
# 117.002381,
# 116.771945,
# 116.746680
# ]


# mean(gpu1)
# mean(gpu2)

# using Pkg
# Pkg.add("StatsPlots")
# using StatsPlots

# cpu = [44.5]
# gpu = [9.04]

# # In PyPlot backend, if we use chars like 'A':'L', ticks are displayed with "PyWrap".
# ticklabel = ["20x20"]
# groupedbar([cpu gpu],
#         bar_position = :stack,
#         bar_width=0.7,
#         xticks=(1:12, ticklabel),
#         label=["cpu" "gpu"]
#         ,size=(300,400)
#         )



# cpu = [2212.1]
# gpu = [119.42]

# # In PyPlot backend, if we use chars like 'A':'L', ticks are displayed with "PyWrap".
# ticklabel = ["40x40"]
# groupedbar([cpu gpu],
#         bar_position = :stack,
#         bar_width=0.7,
#         xticks=(1:12, ticklabel),
#         label=["cpu" "gpu"]
#         ,size=(300,400))




# xs = [x for x in 1:100]
# μs = log.(xs)
# σs = rand(length(xs))

# plot(xs,μs,grid=false,ribbon=σs)

gpu = [9.144303,
8.907645,
8.929718,
9.157349, 
8.887713, 
9.230467,
8.939521,
8.969454,
9.209229
]

cpu = [43.584966,
44.409999,
44.066787,
44.393150,
44.547119,
44.375864,
44.382110,
44.449798,
44.420971]


mean(gpu)
std(gpu)

mean(cpu)
std(cpu)


function box2()
    data = box(;y=cpu,
                name="CPU",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8)
    PlotlyJS.plot(data, Layout(width=400, height=400) )
end
box2()


# function box8()
#     trace1 = box(;y=cpu,
#                   name="CPU")
#     # trace2 = box(;y=gpu,
#     #               name="GPU",
#     #               marker_color="rgb(0, 128, 128)")
#     data = [trace1]
#     layout = Layout(;title="Training and test time")
#     PlotlyJS.plot(data, layout)
# end
# box8()

# function box3()

#     trace1 = box(;x=cpu,
#                   name="CPU")
#     trace2 = box(;x=gpu,
#                   name="GPU",
#                   marker_color="rgb(0, 128, 128)")
#     data = [trace1, trace2]
#     layout = Layout(;title="Horizontal Box Plot")

#     PlotlyJS.plot(data, layout)
# end
# box3()


Plots.bar(["GPU", "CPU"], [9,40] )

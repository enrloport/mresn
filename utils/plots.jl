using Plots

# x = collect(range(0, 2, length= 100))
# y1 = exp.(x)
# y2 = exp.(1.3 .* x)

x = collect(range(-10, 10, length= 100))
function y10(x) return 2 end
y1 = y10.(x)
y2 = 0

plot(x, y10, fillrange = y2, fillalpha = 0.35, c = 1, ylims=(-1,3), label = "Confidence band", legend = :topleft)

plot!([0,0], [-3,3])
plot!([-10, 10],[0,0])
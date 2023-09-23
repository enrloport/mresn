include("../ESN.jl")

_B = 0.3867
_K = 0.6957
_Q = 0.8598
_V = 0.3726

title ="FashionMnist convergence. Seed 42"
label = "B: "*string(_B)*", K: "*string(_K)*", Q: "*string(_Q)*", V: "*string(_V)


#Generalised Logistic Curve
# A: Upper asymptote
# K: Lower asymptote
# C: Typically takes a value of 1. Otherwise, the upper asymptote is A + (K-A) / (C^(1/v))
# B: Growth rate
# v > 0: affects near which asymptote maximum growth occurs
# Q: is related to the value glc(0)
function glc(x; A=-_K, K=_K, C=1.0, B=_B, v=_V, Q=_Q)
    return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/v) )
end

#for i in -10:10
#    v = i
#    println(v," ", glc(v))
#end

x = [x for x in -20:20]
y = [glc(i) for i in x ]
# ys = [ [glc(i/50;B=b/10) for i in x ] for b in 1:max]
plot(x,y; title=title, labels=label  )

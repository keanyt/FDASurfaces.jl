function kernel_gauss(u::Vector{TYPE}, v::Vector{TYPE}, ooSigma2::TYPE) where TYPE
    r2 = norm(u - v)^2
    return exp(-r2 * ooSigma2)
end


function kernel_1d_gauss(u::Vector{TYPE}, v::Vector{TYPE}, ooSigma2::TYPE, l::INT) where TYPE
    r2 = norm(u - v)^2
    return -2 * ooSigma2 * (v[l] - u[l]) * exp(-r2 * ooSigma2)
end


function kernel_gauss_var(u, v, ooSigma2, DIM)
    normu2 = norm(u)^2
    normv2 = norm(v)^2
    prsuv = dot(u, v)
    
    temp = normu2 * normv2
    return √temp * exp(2 * (prsuv^2 / temp) * ooSigma2)
end
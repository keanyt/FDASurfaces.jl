function fPCAManifold(X, mass, stiff, loglambdaseq, n_iter, indices_NAN)

    X_clean = copy(X)
    X_clean[:, indices_NAN] = []
    
    nbasis = size(X, 2)
    
    nlambda = length(loglambdaseq)
    f_hat_mat = zeros(nbasis, nlambda)
    u_hat_mat = zeros(size(X, 1), nlambda)
    
    lambda_values = (10.0.^(loglambdaseq))
    NW = spdiagm(0 => ones(size(mass, 2)))

    for ilambda in 1:nlambda
        
        NW[indices_NAN, indices_NAN] .= 0
        SW = (lambda_values[ilambda]) * stiff
        NE = (lambda_values[ilambda]) * stiff
        SE = -(lambda_values[ilambda]) * mass
        A = [NW NE; SW SE]
        
        U = svd(X_clean, full = true).U
        u_hat = U[:, 1]
        f_vec = zeros(nbasis)

        for iter in 1:n_iter
            
            b = [X' * u_hat; zeros(nbasis)]
            b[indices_NAN] .= 0

            sol = A \ b
            f_vec = sol[1:nbasis]
            
            fp = copy(f_vec)
            fp[indices_NAN] = []
            u_hat = X_clean * fp
            u_hat = u_hat / sqrt(sum(u_hat.^2))
        end
        
        u_hat_mat[:, ilambda] = u_hat
        f_hat_mat[:, ilambda] = f_vec
    end
    
    return f_hat_mat, u_hat_mat
end

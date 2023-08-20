function fPCAManifold_Kfold(X, mass, stiff, loglambdaseq, K, n_iter, indices_NAN)

    #X = x_0
    #mass = R0
    #stiff = R1
    #loglambdaseq = 
    #K = 5
    #n_iter = 20
    #indices_NAN = []


    X_clean = copy(X)
    X_clean[:, indices_NAN] = []
    
    nbasis = size(X, 2)
    nlambda = length(loglambdaseq)

    CVseq = zeros(nlambda)

    lambda_values = (10.0.^(loglambdaseq))
    
    Threads.@threads for ilambda = 1:nlambda

        current_lambda = lambda_values[ilambda]
        
        NW = spdiagm(0 => ones(size(mass, 2)))
        NW[indices_NAN, indices_NAN] .= 0
        SW = (current_lambda) * stiff
        NE = (current_lambda) * stiff
        SE = -(current_lambda) * mass
        A = [NW NE; SW SE]
        
        for ifold = 1:K

            # Split dataset into training set and test set
            length_chunk = floor(Int, size(X, 1)/ K)
            indices_valid = (ifold - 1) * length_chunk .+ (1:length_chunk)
            indices_train = setdiff(1:size(X, 1), indices_valid)

            X_train = X[indices_train, :]
            X_train = X_train - find_ones_mean(X_train)

            X_clean_train = X_clean[indices_train, :]
            X_clean_train = X_clean_train - find_ones_mean(X_clean_train)

            X_clean_valid = X_clean[indices_valid, :]
            X_clean_valid = X_clean_valid - find_ones_mean(X_clean_valid)


            U = svd(X_clean_train).U
            u_hat = U[:, 1]
            fp = 0
            g = 0
            
            for iter = 1:n_iter
                
                b = [X_train' * u_hat; zeros(nbasis)]
                b[indices_NAN] .= 0
                sol = A \ b

                f = sol[1:nbasis]
                fp = copy(f)
                g = sol[(1+nbasis):(2*nbasis)]
                
                fp[indices_NAN] = []
                u_hat = X_clean_train * fp
                u_hat = u_hat / sqrt(sum(u_hat.^2))
            end
            
            u_hat_const = sum(fp .^ 2) + 10.0^(loglambdaseq[ilambda]) * (g' * mass * g)
            u_hat_valid = (X_clean_valid * fp) ./ u_hat_const
            CV_local = min(sum((X_clean_valid - u_hat_valid * fp').^2) /  (size((X_clean_valid),1)*size(X_clean_valid,2)),
                            sum((X_clean_valid + u_hat_valid * fp').^2) /  (size((X_clean_valid),1)*size(X_clean_valid,2)))
            CVseq[ilambda] += CV_local
        end
        println("CV index computed for lambda st log(lambda) = ", loglambdaseq[ilambda])
    end
    
    return CVseq
end
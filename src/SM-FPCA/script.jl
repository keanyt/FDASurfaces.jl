using SparseArrays
using DelimitedFiles
using LinearAlgebra
using CSV
using DataFrames
using Statistics


function find_ones_mean(data_matrix)
    # Calculate column-wise mean using `mean` function
    temp_mean = Statistics.mean(data_matrix, dims = 1)

    # Create a new matrix with each column filled with the corresponding mean value
    return repeat(temp_mean, size(data_matrix, 1), 1)
end

function vector_multiply(vec1, vec2)

    matrix_result = zeros(length(vec1), length(vec2))

    for i=1:length(vec2)
        matrix_result[:,i] = vec1 .* vec2[i]
    end

    return matrix_result
end



X_0 = func_values - find_ones_mean(func_values)

Kfolds = 5
niter = 20
loglambdaseq = -5:5
index_NA = []

N_PC = 5
F_not_normalized = zeros(N_PC, size(X_0, 2))
U_normalized = zeros(size(X_0, 1), N_PC)
optimal_lambdas_indices = zeros(N_PC)

X_residuals = copy(X_0)

for k in 1:N_PC
    println("Computing PC function ", k, "...")
    CVseq = fPCAManifold_Kfold(X_residuals, R0, R1, loglambdaseq, Kfolds, niter, index_NA)
    CV_min_index = argmin(CVseq)
    CV_min = CVseq[CV_min_index]

    println("    Index chosen: ", CV_min_index, ".")
    optimal_lambdas_indices[k] = CV_min_index

    F_not_normalized[k, :], U_normalized[:, k] = fPCAManifold(X_residuals, R0, R1, 10.0^(loglambdaseq[CV_min_index]), niter, index_NA)

    X_residuals = X_residuals - (vector_multiply(U_normalized[:,k], F_not_normalised[k,:]))
end

F_normalized = zeros(size(F_not_normalized))

for k in 1:N_PC
    F_normalized[k, :] = F_not_normalized[k, :] ./ getL2Norm(F_not_normalized[k, :], R0)
end

U_not_normalized = zeros(size(U_normalized))

for k in 1:N_PC
    U_not_normalized[:, k] = U_normalized[:, k] .* getL2Norm(F_not_normalized[k, :], R0)
end


(Q, R) = qr(U_not_normalized)
explained_var = (diag(R) .^ 2) / size(X_0, 1)
println(tr(U_not_normalized' * U_not_normalized))

(U, S, V) = svd(X_0, thin=false)
U_ALL = zeros(size(U_not_normalized))
for i in 1:size(S, 1)
    U_ALL[:, i] = U[:, i] * S[i] * getL2Norm(V[:, i], R0)
end
total_var = trace(U_ALL' * U_ALL) / size(X_0, 1)
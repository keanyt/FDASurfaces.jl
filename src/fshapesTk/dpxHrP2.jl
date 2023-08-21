function gauss_gpu_grad1_conv_on_device!(TYPE::Type, DIMPOINT::int, ooSigma2, alpha, x, beta, gamma, nx)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    shared_data = @cuDynamicSharedMem(TYPE, DIMPOINT * blockDim().x)

    xi = Vector{TYPE}(undef, DIMPOINT)
    xmx = Vector{TYPE}(undef, DIMPOINT)
    alphai = Vector{TYPE}(undef, DIMPOINT)
    gammai = zeros(TYPE, DIMPOINT)

    if i <= nx  # we will compute gammai only if i is in the range
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT + k]
        end
        for k = 1:DIMPOINT
            @inbounds alphai[k] = alpha[(i-1)*DIMPOINT + k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:nx
        j = tile * blockDim().x + threadIdx().x
        if j <= nx  # we load yj and betaj from device global memory only if j <= nx
            inc = 3 * DIMPOINT
            for k = 1:DIMPOINT
                @inbounds shared_data[(threadIdx().x-1)*inc + k] = x[(j-1)*DIMPOINT + k]
            end
            for k = 1:DIMPOINT
                @inbounds shared_data[(threadIdx().x-1)*inc + DIMPOINT + k] = beta[(j-1)*DIMPOINT + k]
            end
            for k = 1:DIMPOINT
                @inbounds shared_data[(threadIdx().x-1)*inc + 2*DIMPOINT + k] = alpha[(j-1)*DIMPOINT + k]
            end
        end
        @synchronize()

        if i <= nx  # we compute gammai only if i is in the range
            xj = shared_data
            betaj = shared_data + DIMPOINT
            alphaj = shared_data + 2*DIMPOINT
            inc = 3 * DIMPOINT

            for jrel = 1:blockDim().x
                if jrel <= nx-jstart+1
                    r2 = 0.0

                    for k = 1:DIMPOINT
                        xmx[k] = xi[k] - xj[k]
                        r2 += xmx[k]*xmx[k]
                    end

                    s = -2.0 * ooSigma2 * exp(-r2*ooSigma2)

                    for k = 1:DIMPOINT
                        t1 = s * betaj[k]

                        for l = 1:DIMPOINT
                            gammai[k] += t1 * xmx[l] * (alphai[l]-alphaj[l])
                        end
                    end

                    xj += inc
                    betaj += inc
                    alphaj += inc
                end
            end
        end
        @synchronize()

        tile += 1
    end

    # save the result in global memory
    if i <= nx
        for k = 1:DIMPOINT
            gamma[(i-1)*DIMPOINT + k] = gammai[k]
        end
    end
end


function gauss_gpu_grad1_conv!(TYPE::type, alpha_h, x_h, beta_h, gamma_h, dimPoint, nx)
    # allocate arrays on device and set values to 0
    x_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    alpha_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    beta_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    gamma_d = CUDA.fill(zero(TYPE), nx * dimPoint)

    # send data from host to device
    CUDA.copyto!(x_d, x_h)
    CUDA.copyto!(alpha_d, alpha_h)
    CUDA.copyto!(beta_d, beta_h)

    # compute on device
    try
        block_size = (192, 1, 1) # CUDA_BLOCK_SIZE in makefile_cuda.sh
        grid_size = (div(nx + block_size[1] - 1, block_size[1]), 1, 1)

        if dimPoint == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 1, ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx)
        elseif dimPoint == 2
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 2, ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx)
        elseif dimPoint == 3
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 3, ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx)
        elseif dimPoint == 4
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 4, ooSigma2, alpha_d, x_d, beta_d, gamma_d, nx)
        else
            println("dpxHrP2 error: dimensions of Gauss kernel not implemented in CUDA")
            CUDA.unsafe_free!(gamma_d)
            CUDA.unsafe_free!(beta_d)
            CUDA.unsafe_free!(alpha_d)
            CUDA.unsafe_free!(x_d)
            return -1
        end

        # block until the device has completed
        CUDA.synchronize()

        # send data from device to host
        CUDA.copyto!(gamma_h, gamma_d)
    
    # ensures memory is freed in case of thrown exception
    finally
        CUDA.unsafe_free!(gamma_d)
        CUDA.unsafe_free!(beta_d)
        CUDA.unsafe_free!(alpha_d)
        CUDA.unsafe_free!(x_d)
    end
    return 0
end
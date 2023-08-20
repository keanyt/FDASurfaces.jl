function gauss_gpu_grad1_conv_on_device!(TYPE::Type, DIMPOINT::int, DIMVECT::int, ooSigma2, alpha, x, y, beta, gamma, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    
    shared_data = @cuDynamicSharedMem(TYPE, (DIMPOINT+DIMVECT) * blockDim().x)

    xi = Vector{TYPE}(undef, DIMPOINT)
    alphai = Vector{TYPE}(undef, DIMVECT)
    xmy = Vector{TYPE}(undef, DIMPOINT)
    gammai = zeros(TYPE, DIMPOINT)

    if i <= nx  # we will compute gammai only if i is in the range
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT + k]
        end
        for k = 1:DIMVECT
            @inbounds alphai[k] = alpha[(i-1)*DIMVECT + k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:ny
        j = tile * blockDim().x + threadIdx().x
        if j <= ny  # we load yj and betaj from device global memory only if j <= ny
            inc = DIMPOINT + DIMVECT
            for k = 1:DIMPOINT
                @inbounds shared_data[(threadIdx().x-1)*inc + k] = y[(j-1)*DIMPOINT + k]
            end
            for k = 1:DIMVECT
                @inbounds shared_data[(threadIdx().x-1)*inc + DIMPOINT + k] = beta[(j-1)*DIMVECT + k]
            end
        end
        @synchronize()

        if i <= nx  # we compute gammai only if i is in the range
            yj = shared_data
            betaj = shared_data + DIMPOINT
            inc = DIMPOINT + DIMVECT

            for jrel = 1:blockDim().x
                if jrel <= ny-jstart+1
                    r2 = 0.0f
                    sga = 0.0f
                    for k = 1:DIMPOINT
                        xmy[k] = xi[k] - yj[k]
                        r2 += xmy[k]*xmy[k]
                    end
                    for k = 1:DIMVECT
                        sga += betaj[k] * alphai[k]
                    end
                    s = expr(-ooSigma2 * 2.0f * sga) * expr(-r2 * ooSigma2)
                    for k = 1:DIMPOINT
                        gammai[k] += s * xmy[k]
                    end
                end
                yj += inc
                betaj += inc
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


function gauss_gpu_grad1_conv!(TYPE::type, alpha_x, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny)
    # allocate arrays on device and set values to 0
    x_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    y_d = CUDA.fill(zero(TYPE), ny * dimPoint)
    alpha_d = CUDA.fill(zero(TYPE), ny * dimVect)
    beta_d = CUDA.fill(zero(TYPE), ny * dimVect)
    gamma_d = CUDA.fill(zero(TYPE), nx * dimVect)

    # send data from host to device
    CUDA.copyto!(x_d, x_h)
    CUDA.copyto!(y_d, y_h)
    CUDA.copyto!(alpha_d, alpha_h)
    CUDA.copyto!(beta_d, beta_h)

    # compute on device
    try
        block_size = (192, 1, 1) # CUDA_BLOCK_SIZE in makefile_cuda.sh
        grid_size = (div(nx + block_size[1] - 1, block_size[1]), 1, 1)

        if dimPoint == 1 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 1, 1, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 2 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 2, 1, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 3 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 3, 1, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 4 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 4, 1, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 2 && dimVect == 2
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 2, 2, ooSigma2,alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 3 && dimVect == 3
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 3, 3, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 4 && dimVect == 4
            @cuda threads=block_size blocks=grid_size gauss_gpu_grad1_conv_on_device!(TYPE, 4, 4, ooSigma2, alpha_d, x_d, y_d, beta_d, gamma_d, nx, ny)
        else
            println("gauss_gpu_grad1_conv error: dimensions of Gauss kernel not implemented in CUDA")
            CUDA.unsafe_free!(gamma_d)
            CUDA.unsafe_free!(beta_d)
            CUDA.unsafe_free!(alpha_d)
            CUDA.unsafe_free!(y_d)
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
        CUDA.unsafe_free!(y_d)
        CUDA.unsafe_free!(x_d)
    end
    return 0
end


function exit_function()
    CUDA.reset()
end
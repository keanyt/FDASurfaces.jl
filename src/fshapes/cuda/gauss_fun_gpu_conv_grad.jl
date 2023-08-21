function gauss_gpu_conv_on_device!(TYPE::Type, DIMPOINT::Int, DIMVECT::Int, ooSigmax2, ooSigmaf2, x, y, beta, gamma, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    shared_data = @cuDynamicSharedMem(TYPE, (DIMPOINT+DIMVECT) * blockDim().x)

    xi = Vector{TYPE}(undef, DIMPOINT)
    gammai = zeros(TYPE, (DIMPOINT-1)*DIMVECT)

    if i <= nx  # we will compute gammai only if i is in the range
        # load xi from device global memory
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT + k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:ny
        j = tile * blockDim().x + threadIdx().x
        if j <= ny  # we load yj and betaj from device global memory only if j<ny
            inc = DIMPOINT + DIMVECT
            for k = 1:DIMPOINT
                @inbounds shared_data[(threadIdx().x-1) * inc + k] = y[(j-1) * DIMPOINT + k]
            end
            for k = 1:DIMVECT
                @inbounds shared_data[(threadIdx().x-1) * inc + DIMPOINT + k] = beta[(j-1) * DIMVECT + k]
            end
        end
        @synchronize()

        if i <= nx  # we compute gammai only if needed
            yj = shared_data
            betaj = shared_data + DIMPOINT
            inc = DIMPOINT + DIMVECT

            for jrel = 1:blockDim().x
                if jrel <= ny-jstart+1
                    rx2 = 0.0
                    rf2 = 0.0
                    ximyj = Vector{TYPE}(undef, DIMPOINT-1)

                    for k = 1:(DIMPOINT-1)
                        ximyj[k] = xi[k]-yj[k]
                        rx2 += ximyj[k]*ximyj[k]
                    end

                    fimgj = xi[k]-yj[k]
                    rf2 += fimgj*fimgj
                    s = -2 * ooSigmaf2 * expr(-rx2*ooSigmax2-rf2*ooSigmaf2)

                    for l = 1:DIMVECT
                        for k = 1:(DIMPOINT-1)
                            @inbounds gammai[l+k*DIMVECT] += ximyj[k] * s * betaj[l]
                        end
                    end

                    yj += inc
                    betaj += inc
                end
            end
        end
        @synchronize()

        tile += 1
    end

    # save the result in global memory
    if i <= nx
        for k = 1:(DIMPOINT-1)
            for l = 1:DIMVECT
                @inbounds gamma[(i-1)*DIMVECT + k + (DIMVECT)*nx*l] = gammai[l*DIMVECT + k]
            end
        end
    end
end


function gauss_gpu_eval_conv!(TYPE::type, ooSigmax2, ooSigmaf2, x_h, y_h, beta_h, gamma_h, dimPoint, dimVect, nx, ny)
    # allocate arrays on device and set values to 0
    x_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    y_d = CUDA.fill(zero(TYPE), ny * dimPoint)
    beta_d = CUDA.fill(zero(TYPE), ny * dimVect)
    gamma_d = CUDA.fill(zero(TYPE), nx * dimVect)

    # send data from host to device
    CUDA.copyto!(x_d, x_h)
    CUDA.copyto!(y_d, y_h)
    CUDA.copyto!(beta_d, beta_h)

    # compute on device
    try
        block_size = (192, 1, 1) # CUDA_BLOCK_SIZE in makefile_cuda.sh
        grid_size = (div(nx + block_size[1] - 1, block_size[1]), 1, 1)

        if dimPoint == 2 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 2, 1, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 3 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 3, 1, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 4 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 4, 1, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 5 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 5, 1, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 6 && dimVect == 1
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 6, 1, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 3 && dimVect == 2
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 3, 2, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        elseif dimPoint == 4 && dimVect == 3
            @cuda threads=block_size blocks=grid_size gauss_gpu_conv_on_device!(TYPE, 4, 3, ooSigmax2, ooSigmaf2, x_d, y_d, beta_d, gamma_d, nx, ny)
        else
            println("gauss_fun_gpu_conv_grad_fun error: dimensions of Gauss kernel not implemented in CUDA")
            CUDA.unsafe_free!(gamma_d)
            CUDA.unsafe_free!(beta_d)
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
        CUDA.unsafe_free!(y_d)
        CUDA.unsafe_free!(x_d)
    end
    return 0
end
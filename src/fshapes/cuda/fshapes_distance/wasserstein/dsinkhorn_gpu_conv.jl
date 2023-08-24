include("sinkhorn_gpu_utils.jl")

function sinkhorn_gpu_grad(TYPE::Type, epsilon, lambda, weightGeom, weightGrass, 
                            alpha_h, x_h, y_h, beta_h, mu_h, nu_h, gammax_h, gammadx_h, 
                            dimPoint, dimVect, nx, ny, max_iter)
    alpha_d = CUDA.fill(zero(TYPE), nx * dimVect)
    x_d = CUDA.fill(zero(TYPE), nx * dimPoint)
    y_d = CUDA.fill(zero(TYPE), ny * dimPoint)
    beta_d = CUDA.fill(zero(TYPE), ny * dimVect)
    mu_d = CUDA.fill(zero(TYPE), nx)
    nu_d = CUDA.fill(zero(TYPE), ny)
    gammax_d = CUDA.fill(zero(TYPE), nx * dimVect)
    gammay_d = CUDA.fill(zero(TYPE), ny * dimVect)
    gammadx_d = CUDA.fill(zero(TYPE), nx * dimPoint)

    CUDA.copyto!(x_d, x_h)
    CUDA.copyto!(y_d, y_h)
    CUDA.copyto!(mu_d, mu_h)
    CUDA.copyto!(nu_d, nu_h)

    CUDA.copyto!(alpha_d, alpha_h)
    CUDA.copyto!(beta_d, beta_h)

    # compute on device
    try
        block_size_x = (192, 1, 1) # CUDA_BLOCK_SIZE in makefile_cuda.sh
        grid_size_x = (div(nx + block_size_x[1] - 1, block_size_x[1]), 1, 1)

        block_size_y = (192, 1, 1) # CUDA_BLOCK_SIZE in makefile_cuda.sh
        grid_size_y = (div(ny + block_size_y[1] - 1, block_size_y[1]), 1, 1)

        for iter = 1:max_iter
            if dimpoint == 2 && dimVect == 1
                @cuda threads=block_size_x blocks=grid_size_x sinkhorn_gpu_grad_conv_on_device(TYPE, 2, 1, epsilon, lambda, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny)
            elseif dimPoint == 3 && dimVect == 1
                @cuda threads=block_size_x blocks=grid_size_x sinkhorn_gpu_grad_conv_on_device(TYPE, 3, 1, epsilon, lambda, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny)
            elseif dimPoint == 4 && dimVect == 1
                @cuda threads=block_size_x blocks=grid_size_x sinkhorn_gpu_grad_conv_on_device(TYPE, 4, 1, epsilon, lambda, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny)
            elseif dimPoint == 6 && dimVect == 1
                @cuda threads=block_size_x blocks=grid_size_x sinkhorn_gpu_grad_conv_on_device(TYPE, 6, 1, epsilon, lambda, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny)
            elseif dimPoint == 8 && dimVect == 1
                @cuda threads=block_size_x blocks=grid_size_x sinkhorn_gpu_grad_conv_on_device(TYPE, 8, 1, epsilon, lambda, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, mu_d, gammax_d, nx, ny)
            else
                println("dsinkhorn_gpu_conv error: dimensions of sinkhorn_gpu_grad_conv_on_device kernel not implemented in CUDA")
                CUDA.unsafe_free!(gammay_d)
                CUDA.unsafe_free!(gammax_d)
                CUDA.unsafe_free!(nu_d)
                CUDA.unsafe_free!(mu_d)
                CUDA.unsafe_free!(beta_d)
                CUDA.unsafe_free!(y_d)
                CUDA.unsafe_free!(x_d)
                CUDA.unsafe_free!(alpha_d)
                return -1
            end

            # update u
            if dimpoint == 2 && dimVect == 1
                @cuda threads=block_size_y blocks=grid_size_y sinkhorn_gpu_grad_conv_on_device(TYPE, 2, 1, epsilon, lambda, weightGeom, weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx)
            elseif dimpoint == 3 && dimVect == 1
                @cuda threads=block_size_y blocks=grid_size_y sinkhorn_gpu_grad_conv_on_device(TYPE, 3, 1, epsilon, lambda, weightGeom, weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx)
            elseif dimpoint == 4 && dimVect == 1
                @cuda threads=block_size_y blocks=grid_size_y sinkhorn_gpu_grad_conv_on_device(TYPE, 4, 1, epsilon, lambda, weightGeom, weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx) 
            elseif dimpoint == 6 && dimVect == 1
                @cuda threads=block_size_y blocks=grid_size_y sinkhorn_gpu_grad_conv_on_device(TYPE, 6, 1, epsilon, lambda, weightGeom, weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx) 
            elseif dimpoint == 8 && dimVect == 1
                @cuda threads=block_size_y blocks=grid_size_y sinkhorn_gpu_grad_conv_on_device(TYPE, 8, 1, epsilon, lambda, weightGeom, weightGrass, beta_d, y_d, x_d, alpha_d, nu_d, gammay_d, ny, nx)
            else
                println("dsinkhorn_gpu_conv error: dimensions of sinkhorn_gpu_grad_conv_on_device kernel not implemented in CUDA")
                CUDA.unsafe_free!(gammay_d)
                CUDA.unsafe_free!(gammax_d)
                CUDA.unsafe_free!(nu_d)
                CUDA.unsafe_free!(mu_d)
                CUDA.unsafe_free!(beta_d)
                CUDA.unsafe_free!(y_d)
                CUDA.unsafe_free!(x_d)
                CUDA.unsafe_free!(alpha_d)
                return -1
            end
            
            # update v
            CUDA.copyto!(beta_d, gammay_d)
        end

        # block until the device has completed
        CUDA.synchronize()

        # send data from device to host
        CUDA.copyto!(gammax_h, alpha_d)

        # compute the gradient of the energy
        if dimpoint == 2 && dimVect == 1
            @cuda threads=block_size_y blocks=grid_size_y grad_xW_on_device(TYPE, 2, 1, epsilon, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, gammadx_d, nx, ny)
        elseif dimpoint == 3 && dimVect == 1
            @cuda threads=block_size_y blocks=grid_size_y grad_xW_on_device(TYPE, 3, 1, epsilon, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, gammadx_d, nx, ny)
        elseif dimpoint == 4 && dimVect == 1
            @cuda threads=block_size_y blocks=grid_size_y grad_xW_on_device(TYPE, 4, 1, epsilon, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, gammadx_d, nx, ny)
        elseif dimpoint == 6 && dimVect == 1
            @cuda threads=block_size_y blocks=grid_size_y grad_xW_on_device(TYPE, 6, 1, epsilon, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, gammadx_d, nx, ny)
        elseif dimpoint == 8 && dimVect == 1
            @cuda threads=block_size_y blocks=grid_size_y grad_xW_on_device(TYPE, 8, 1, epsilon, weightGeom, weightGrass, alpha_d, x_d, y_d, beta_d, gammadx_d, nx, ny)
        else
            println("dsinkhorn_gpu_conv error: dimensions of sinkhorn_gpu_grad_conv_on_device kernel not implemented in CUDA")
            CUDA.unsafe_free!(gammay_d)
            CUDA.unsafe_free!(gammax_d)
            CUDA.unsafe_free!(nu_d)
            CUDA.unsafe_free!(mu_d)
            CUDA.unsafe_free!(beta_d)
            CUDA.unsafe_free!(y_d)
            CUDA.unsafe_free!(x_d)
            CUDA.unsafe_free!(alpha_d)
            return -1
        end

        # block until the device has completed
        CUDA.synchronize()

        # send data from device to host
        CUDA.copyto!(gammadx_h, gammadx_d)
    
    # ensures memory is freed in case of thrown exception
    finally
        CUDA.unsafe_free!(gammay_d)
        CUDA.unsafe_free!(gammax_d)
        CUDA.unsafe_free!(nu_d)
        CUDA.unsafe_free!(mu_d)
        CUDA.unsafe_free!(beta_d)
        CUDA.unsafe_free!(y_d)
        CUDA.unsafe_free!(x_d)
        CUDA.unsafe_free!(alpha_d)
    end
    return 0
end
using CUDA

# set the default UseCudaOnDoubles instead of Floats
EPSILON = 2.220446049250313e-16
# EPSILON	= 1.1920929e-07

#*********************************************************#
#                 Fast Inverse Square Root                #
#*********************************************************#

function invsqrt(x::Float32)
    y = x
    i = reinterpret(UInt32, y)
    i = 0x5f3759df - i >> 1     # Quake rsqrt magic number
    y = reinterpret(Float32, i)
    y *= 1.5 - 0.5*x*(y^2)
    return y
end


# based on https://stackoverflow.com/questions/11644441/fast-inverse-square-root-on-x64/11644533#11644533
function invsqrt(x::Float64)
    y = x
    i = reinterpret(UInt64, y)
    # this magic number is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    i = 0x5fe6eb50c7b537a9 - (i >> 1)
    y = reinterpret(Float64, i)
    y *= 1.5 - 0.5*x*(y^2)
    return y
end

#*********************************************************#
#                     Cost functions                      #
#*********************************************************#

# do we need TYPE and DIMPOINT here???
function cost_ptcloud(TYPE::Type, DIMPOINT::int, DIMVECT::int,
                        alphai, xi, yj, betaj, weightGeom, epsilon, gammai)
    # for k = 1:DIMPOINT
    #     ximxj = xi[k]-yj[k]
    #     r2 += ximxj * ximxj
    # end
    r2 = norm(xi - yj)^2 * 0.5 * weightGeom

    for k = 1:DIMVECT
        s = exp((-r2 + alphai[k] + betaj[k]) / epsilon)
        gammai[k] += s
    end
end


function dcost_ptcloud(TYPE::Type, DIMPOINT::int, DIMVECT::int,
                        alphai, xi, yj, betaj, weightGeom, epsilon, gammai)
    ximxj = Vector{TYPE}(undef, DIMPOINT)
    r2 = norm(xi - yj)^2 * 0.5 * weightGeom
    s = 0.0

    for k = 1:DIMPOINT
        ximxj[k] = xi[k]-yj[k]
    end
    for k = 1:DIMVECT
        s += expr((-r2 + alphai[k] + betaj[k])/epsilon)
    end
    for k = 1:DIMPOINT
        gammai[k] += weightGeom * ximxj[k]*s
    end
end


function cost_varifold(TYPE::Type, DIMPOINT::int, DIMVECT::int,
                        alphai, xi, yj, betaj, weightGeom, weightGrass, epsilon, gammai)
    r2 = norm(xi -yj)^2 * 0.5 * weightGeom
    s = 0.0
    normxii = 0.0
    normxij = 0.0
    prsxiixj = 0.0

    for k = 1:DIMPOINT/2
        normxii += xi[k+DIMPOINT/2] * xi[k+DIMPOINT/2]
        normxij += yj[k+DIMPOINT/2] * yj[k+DIMPOINT/2]
        prsxiixj += xi[k+DIMPOINT/2] * yj[k+DIMPOINT/2]
    end

    # unoriented
    r2 += weightGrass * (2 - (2 * prsxiixj*prsxiixj * invsqrt(normxij * normxii)))
    # oriented
    # r2 += weightGrass * (1 - (prsxiixj * invsqrt(normxij * normxii)))

    for k = 1:DIMVECT
        s += expr((-r2 + alphai[k] + betaj[k])/epsilon)
        gammai[k] += s
    end
end


function dcost_varifold(TYPE::Type, DIMPOINT::int, DIMVECT::int,
                        alphai, xi, yj, betaj, weightGeom, weightGrass, epsilon, gammai)
    ximxj = Vector{TYPE}(undef, DIMPOINT)
    r2 = norm(xi -yj)^2 * 0.5 * weightGeom
    s = 0.0
    normxii = 0.0
    normxij = 0.0
    prsxiixj = 0.0

    for k = 1:DIMPOINT/2
        ximxj[k] = xi[k]-yj[k]
        normxii += xi[k+DIMPOINT/2] * xi[k+DIMPOINT/2]
        normxij += yj[k+DIMPOINT/2] * yj[k+DIMPOINT/2]
        prsxiixj += xi[k+DIMPOINT/2] * yj[k+DIMPOINT/2]
    end

    # unoriented
    r2 += weightGrass * (2 - (2 * prsxiixj*prsxiixj * invsqrt(normxij * normxii)))
    # oriented
    # r2 += weightGrass * (1 - (prsxiixj * invsqrt(normxij * normxii)))

    for k = 1:DIMVECT
        s += expr((-r2 + alphai[k] + betaj[k])/epsilon)
    end

    for k = 1:DIMPOINT/2
        gammai[k] += weightGeom * ximxj[k] * s
        # unoriented
        gammai[k+DIMPOINT/2] -= 8 * prsxiixj * weightGrass * (yj[k+DIMPOINT/2] - prsxiixj * xi[k+DIMPOINT/2] / normxii) * invsqrt(normxij*normxii) * s
        # oriented
        # gammai[k+DIMPOINT/2] -= weightGrass * (yj[k+DIMPOINT/2] - prsxiixj * xi[k+DIMPOINT/2] / normxii) * invsqrt(normxij+normxii) * s
    end
end

#*********************************************************#
#                 A Single Sinkhorn Step                  #
#*********************************************************#

function sinkhorn_gpu_grad_conv_on_device(TYPE::type, DIMPOINT::int, DIMVECT::int,
                                            epsilon, lambda, weightGeom, weightGrass, alpha, x, y, beta, delta, gamma, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    # or is it (DIMPOINT * DIMVECT)?
    shared_data = @cuDynamicSharedMem(TYPE, (DIMPOINT + DIMVECT) * blockDim().x)
    inc = DIMPOINT + DIMVECT

    xi = Vector{TYPE}(undef, DIMPOINT)
    alphai = Vector{TYPE}(undef, DIMVECT)
    deltai = Vector{TYPE}(undef, DIMVECT)
    gammai = zeros(TYPE, DIMVECT)

    if i <= nx
        # load xi, alphai, betai from device global memory (???)
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT+k]
        end
        for k = 1:DIMVECT
            @inbounds alphai[k] = alpha[(i-1)*DIMVECT+k]
        end
        for k = 1:DIMVECT
            @inbounds deltai[k] = delta[(i-1)*DIMVECT+k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:ny
        j = tile * blockDim().x + threadIdx().x
        if j <= ny  # we load yj and betaj from device global memory only if j <= ny
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
            for jrel = 1:blockDim().x
                if jrel <= nx-jstart+1
                    if weightGrass == 0
                        cost_ptcloud(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, epsilon, gammai)
                    else
                        cost_varifold(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, weightGrass, epsilon, gammai)
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
        si = 0.0
        for k = 1:DIMVECT
            si += lambda * (epsilon * log(deltai[k]) - epsilon * log(gammai[k] + EPSILON) + alphai[k])
        end
        gamma[i] = si;
    end
end

#*********************************************************#
#                      Compute Wdual                      #
#*********************************************************#

function Wdual_on_device(TYPE::type, DIMPOINT::int, DIMVECT::int,
                            epsilon, weightGeom, weightGrass, alpha, x, y, beta, gamma, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    shared_data = @cuDynamicSharedMem(TYPE, (DIMPOINT + DIMVECT) * blockDim().x)
    inc = DIMPOINT + DIMVECT

    xi = Vector{TYPE}(undef, DIMPOINT)
    alphai = Vector{TYPE}(undef, DIMVECT)
    gammai = zeros(TYPE, DIMVECT)

    if i <= nx
        # load xi, alphai, betai from device global memory (???)
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT+k]
        end
        for k = 1:DIMVECT
            @inbounds alphai[k] = alpha[(i-1)*DIMVECT+k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:ny
        j = tile * blockDim().x + threadIdx().x
        if j <= ny  # we load yj and betaj from device global memory only if j <= ny
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
            for jrel = 1:blockDim().x
                if jrel <= nx-jstart+1
                    if weightGrass == 0
                        cost_ptcloud(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, epsilon, gammai)
                    else
                        cost_varifold(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, weightGrass, epsilon, gammai)
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
        si = 0.0
        for k = 1:DIMVECT
            si += -epsilon * gammai[k]
        end
        gamma[i] = si;
    end
end

#*********************************************************#
#                      Compute grad_x                     #
#*********************************************************#

function grad_xW_on_device(TYPE::type, DIMPOINT::int, DIMVECT::int,
                            epsilon, weightGeom, weightGrass, alpha, x, y, beta, gamma, nx, ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    shared_data = @cuDynamicSharedMem(TYPE, (DIMPOINT + DIMVECT) * blockDim().x)
    inc = DIMPOINT + DIMVECT

    xi = Vector{TYPE}(undef, DIMPOINT)
    alphai = Vector{TYPE}(undef, DIMVECT)
    gammai = zeros(TYPE, DIMPOINT)

    if i <= nx
        # load xi, alphai, betai from device global memory (???)
        for k = 1:DIMPOINT
            @inbounds xi[k] = x[(i-1)*DIMPOINT+k]
        end
        for k = 1:DIMVECT
            @inbounds alphai[k] = alpha[(i-1)*DIMVECT+k]
        end
    end

    tile = 0
    for jstart = 1:blockDim().x:ny
        j = tile * blockDim().x + threadIdx().x
        if j <= ny  # we load yj and betaj from device global memory only if j <= ny
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
            for jrel = 1:blockDim().x
                if jrel <= nx-jstart+1
                    if weightGrass == 0
                        dcost_ptcloud(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, epsilon, gammai)
                    else
                        dcost_varifold(TYPE, DIMPOINT, DIMVECT, alphai, xi, yj, betaj, weightGeom, weightGrass, epsilon, gammai)
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
        for k = 1:DIMPOINT
            gamma[(i-1)*DIMPOINT + k] = gammai[k]
        end
    end
end


function exit_function()
    CUDA.reset()
end
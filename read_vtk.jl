struct fshape
    v::Array{Float64, 2}
    f::Array{Int64, 2}
    s::Array{Float64, 1}
end

function import_fshape_vtk_femurs(filename)
    
    v, f, s = read_vtk(filename)
    return fshape(v, f, s)
end

function read_vtk(filename)
    
    fid = open(filename, "r")
    
    # Read header
    str = readline(fid)
    if !startswith(str, "# vtk")
        error("The file is not a valid VTK one.")
    end
    
    # Skip 3 lines
    for _ in 1:3
        readline(fid)
    end
    
    # Read vertices
    str = readline(fid)
    info = match(r"POINTS (\d+)", str)
    if info === nothing
        error("Problem in reading vertices.")
    end

    n_points = parse(Int,info.captures[1])
    vertex = zeros(Float64, (n_points,3))
    vertex_count = 1

    n_lines = floor(Int, n_points/3)

    for i in 1:(n_lines)
        
        simplex_coords = split(readline(fid), " ")
        simplex_coords = simplex_coords[1:9]
        simplex_coords = parse.(Float64, simplex_coords)
        simplex_coords = reshape(simplex_coords, (3,3))'
        
        vertex[vertex_count,:] = simplex_coords[1,:]
        vertex[vertex_count+1,:] = simplex_coords[2,:]
        vertex[vertex_count+2,:] = simplex_coords[3,:]

        vertex_count = vertex_count + 3
    end

    println("Vertices found.")
    
    readline(fid)
    # Read polygons or lines
    str = readline(fid)
    info = match(r"(POLYGONS|LINES) (\d+) (\d+)", str)
    if info === nothing
        error("Problem in reading faces.")
    end

    n_face = parse.(Int, info.captures[2:end])[1]
    polygons = zeros(Int, (n_face,3))

    for i in 1:(n_face)

        polygon_data = split(readline(fid), " ")
        polygon_data = polygon_data[2:4]
        polygon_data = parse.(Int, polygon_data) .+ 1

        polygons[i,:] = polygon_data
    end

    println("Polygons found.")
    # Read signal
    signal = zeros(Float64, n_points)
    field_found = false

    while !field_found && !eof(fid)
        str = readline(fid)

        if contains(str, "FIELD")
            readline(fid)
            field_found = true   
        end
    end


    signal_count = 1

    while !eof(fid)

        field_data = split(readline(fid)," ")
        field_data = field_data[1:(length(field_data)-1)]
        field_data = parse.(Float64, field_data) 

        signal[signal_count:(signal_count+length(field_data)-1)] = field_data
        signal_count = signal_count + length(field_data)
    end

    println("Signals found.")
    
    close(fid)
    return vertex, polygons, signal
end
    


    
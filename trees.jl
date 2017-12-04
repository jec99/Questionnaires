

using Clustering

#=

definining a partition tree type --- a hiersrchy of folders
for now, each folder in a partition tree only contains indices,
    not points or other values. it's assume that if the user
    wants to access this information they can do so externally;
    the tree is only here for the tree structure

=#

mutable struct Folder
    parent::Folder  # folder or nothing
    children::Array{Folder,1}
    inds::IntSet

    function Folder(inds)
        x = new()
        x.children = Folder[]
        x.inds = IntSet(inds)
        return x
    end
    Folder() = Folder([])
end


mutable struct Partition
    folders::Array{Folder,1}

    Partition(folders) = (x = new(); x.folders = folders; x)
    Partition() = Partition([])
end


mutable struct PartitionTree
    #=
    root has as its index set the entire found set, and the leaves
    have all singletons
    =#
    ground_set::IntSet
    levels::Array{Partition,1}
end

function PartitionTree(ground_set; make_leaves=false)
    ground_set = IntSet(ground_set)
    root = Folder(ground_set)
    root_p = Partition([root])
    tree = PartitionTree(ground_set,[root_p])
    if make_leaves
        leaves = Partition()
        push!(tree.levels,leaves)
        for i in ground_set
            leaf = Folder(root,[],IntSet([i]))
            leaf.parent = root
            push!(root.children,leaf)
            push!(leaves.folders,leaf)
        end
    end
    return tree
end


function correlation_similarity(data::Array{Number,2},part::Partition; normalize_=true,center_=true)
    #=
    - computes the correlations of (the columns of) data, under a partition
    of the rows (features) of data
    - returns a similarity matrix

    corresponds to partition_localgeometry in coifman's code
    =#

    n_features, n_points = size(data)
    if n_features != len(part.gound_set)
        error('partition must be of the rows of data')
    end

    W = zeros(n_points,n_points)
    total_len = 0
    for folder in part.folders
        inds = collect(folder.inds)
        if length(inds) == 1
            continue
        end
        total_len += length(inds)
        sub_data = data[inds,:]

        if center_
            sub_data = sub_data .- mean(sub_data,1)
        end
        if normalize_
            sub_data = sub_data ./ sqrt(sum(abs2,sub_data,1))
        end

        # dense matrix arithmetic; maybe sparsify later
        # with a nearest neighbor search
        W += abs.(sub_data'*conj.(sub_data))*length(inds);
    end

    if total_len > 0
        W ./= total_len
    end

    return W
end


function kmeans_rep(data,k;replications=10)
    best = nothing
    best_cost = Inf
    for _ in 1:replications
        a = kmeans(data,k)
        if a.totalcost < best_cost
            best_cost = a.totalcost
            best = a.assignments
        end
    end
    return best
end


function dyadic_tree(vecs,vals; max_levels=12,n_replicates=10)
    #=
    maxvals=0.99;
    NLevels = 12;

    takes in eigenvectors and eigenvalues coming from a similarity
    matrix (an embedding) and returns a partition tree of the
    points they parameterize based on the geometry of this embedding

    vecs should be d x n, where d is the embedding dimension and n
    is the number of points. these are meant to be the top eigenvectors
    and eigenvalues of a similarity matrix
    =#

    if std(vecs[1,:]) == 0
        vecs = vecs[2:end,:]
        vals = vals[2:end]
    end
    d,n_points = size(vecs)
    levels = min(levels,d)
    vecs = vecs .* vals

    tree = PartitionTree(1:n_points)
    parents = tree.levels[1]
    for l in 2:max_levels
        part = Partition()
        max_folder_size = 0

        # subdivide folders
        for folder in parents
            idxs = collect(folder.idxs)
            sz = length(idxs)
            max_folder_size = max(max_folder_size,sz)
            sub_data = vecs[:,idxs]'

            if sz >= 4
                a = kmeans_rep(sub_data,k)
            else  # if the folder is very small, don't break it up
                a = Array{Int}(ones(sz));
            end

            child1 = Folder(folder,[],idxs[a .== 1])
            push!(part,child1)
            push!(folder.children,child1)
            if length(unique(a)) == 2
                child2 = Folder(folder,[],idxs[a .== 2])
                push!(part,child2)
                push!(folder.children,child2)
            end
        end

        push!(tree.levels,part)
        parents = part

        if max_folder_size < 4
            break
        end
    end

    # take care of leaves
    leaves = Partition()
    for folder in parents
        for i in folder.idxs
            child = Folder(folder,[],[i])
            push!(leaves,child)
            push!(folder.children,child)
        end
    end
    push!(tree.levels,leaves)

    return tree
end


function dual_geometry(data,tree; alpha=1,normalize_=true,center_=true)
    #=
    computes the geometry (in the form of a similarity matrix) of a dataset
    based on a geometry of its features (rows). this can then be used to
    construct a tree on the data points (columns)
    =#

    n_features, n_points = size(data)
    depth = length(tree.levels)
    W = np.zeros(n_points,n_points)
    # don't interate over root or leaves
    for i in 2:depth-1
        similarity = correlation_similarity(data,tree.levels[i],normalize_=normalize_,center=center)
        similarity = similarity + similarity'
        W += 2^(-alpha*i)*similarity
        # this maybe not be exactly right, unless the folder divisions are balanced
    end

    return W
end




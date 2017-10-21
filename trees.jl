
#=

definining a partition tree type --- a hiersrchy of folders
for now, each folder in a partition tree only contains indices,
    not points or other values. it's assume that if the user
    wants to access this information they can do so externally;
    the tree is only here for the tree structure

=#

mutable struct Folder
    parent::Nullable{Folder}  # folder or nothing
    children::Array{Folder,1}
    inds::IntSet
end

Folder() = Folder(nothing,[],[])

Folder(inds) = Folder(nothing,[],IntSet(inds))


mutable struct Partition
    folders::Array{Folder,1}
end

Partition() = Partition([])


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

    corresponds to parition_localgeometry in coifman's code
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
            sub_data = sub_data ./ sum(abs2,sub_data,1)
        end

        # dense matrix arithmetic; maybe sparsify later
        # with a nearest neighbor search
        W += abs.(sub_data'*conj.(sub_data))*length(inds);
    end

    if total_len >
        W ./= total_len
    end

    return W
end


function dyadic_tree(vecs,vals; levels=12,n_replicates=10)
    #=
    maxvals=0.99;
    NLevels = 12;

    takes in eigenvectors and eigenvalues coming from a similarity
    matrix (an embedding) and returns a partition tree of the
    points they parameterize based on the geometry of this embedding

    vecs should be d x n, where d is the embedding dimension and n
    is the number of points
    =#

    if std(vecs[1,:]) == 0
        vecs = vecs[2:end,:]
        vals = vals[2:end]
    end
    d, n_points = size(vecs)
    levels = min(levels,d)
    vecs = vecs .* vals

    tree = PartitionTree(1:n_points)
    parents = tree.levels[1]
    for l in 1:levels
        part = Partition()
        max_folder_size = 0
        # TODO: FINISH
    end
end
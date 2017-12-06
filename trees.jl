

using Clustering
using Base

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
    idxs::IntSet

    function Folder(idxs)
        x = new()
        x.children = Folder[]
        x.idxs = IntSet(idxs)
        return x
    end
    Folder() = Folder([])
    function Folder(p,c,idxs)
        x = new()
        x.parent = p
        x.children = c
        x.idxs = IntSet(idxs)
        return x
    end
end

function str(f::Folder; name_=true)
    if name_
        return join(["Folder([",join(collect(f.idxs),","),"])"])
    else
        return join(["[",join(collect(f.idxs),","),"]"])
    end
end

function Base.show(io::IO,f::Folder)
    print(io,str(f))
end


mutable struct Partition
    folders::Array{Folder,1}
    ground_set::IntSet

    function Partition(folders::Array{Folder,1})
        x = new();
        x.folders = folders;
        x.ground_set = union([f.idxs for f in folders]...)
        return x
    end
    function Partition(idxs)
        # this is somewhat lazy programming but this takes in a collection
        # of integers of some sort - anything that can be casted to IntSet
        x = new();
        x.folders = [Folder(idxs)];
        x.ground_set = IntSet(idxs);
        return x
    end
    Partition() = (x = new(); x.folders = []; x.ground_set = IntSet(); x)
end

function Base.show(io::IO,p::Partition)
    to_print = join(["Partition(",join([str(f,name_=false) for f in p.folders],","),")"])
    print(io,to_print)
end

mutable struct PartitionTree
    #=
    root has as its index set the entire found set, and the leaves
    have all singletons
    =#
    ground_set::IntSet
    levels::Array{Partition,1}
end

function Base.show(io::IO,t::PartitionTree)
    print(t.levels[1])
    for l in t.levels[2:end]
        print(io,"\n");print(io,l)
    end
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
            leaf = Folder(root,Folder[],IntSet([i]))
            leaf.parent = root
            push!(root.children,leaf)
            push!(leaves.folders,leaf)
        end
    end
    return tree
end


function correlation_similarity(data,part::Partition; normalize_=true,center_=true,cutoff=1)
    #=
    - computes the correlations of (the columns of) data, under a partition
    of the rows (features) of data
    - returns a similarity matrix

    corresponds to partition_localgeometry in coifman's code
    =#

    n_features, n_points = size(data)
    if n_features != length(part.ground_set)
        error("partition must be of the rows of data")
    end

    W = zeros(n_points,n_points)
    total_len = 0
    for folder in part.folders
        idxs = collect(folder.idxs)
        if length(idxs) <= cutoff
            continue
        end
        total_len += length(idxs)
        sub_data = data[idxs,:]

        if center_
            sub_data = sub_data .- mean(sub_data,1)
        end
        if normalize_
            sub_data = sub_data ./ sqrt.(sum(abs2,sub_data,1))
        end

        # dense matrix arithmetic; maybe sparsify later
        # with a nearest neighbor search
        W += abs.(sub_data'*conj.(sub_data))*length(idxs);
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

    eigenvalues coming from a similarity
    matrix (an embedding) and returns a partition tree of the
    points they parameterize based on the geometry of this embedding

    vecs should be n x d, where d is the embedding dimension and n
    is the number of points. these are meant to be the top eigenvectors
    and eigenvalues of a similarity matrix
    =#

    # removed because redundant
    # if std(vecs[:,1]) == 0
    #     vecs = vecs[:,2:end]
    #     vals = vals[2:end]
    # end

    n_points,d = size(vecs)
    levels = min(max_levels,d)
    vecs = vecs .* vals'

    tree = PartitionTree(1:n_points)
    parents = tree.levels[1]
    for l in 2:max_levels
        part = Partition()
        max_folder_size = 0

        # subdivide folders
        for folder in parents.folders
            idxs = collect(folder.idxs)
            sz = length(idxs)
            max_folder_size = max(max_folder_size,sz)
            sub_data = vecs[idxs,:]

            if sz >= 4
                a = kmeans_rep(sub_data',2,replications=n_replicates)
            else  # if the folder is very small, don't break it up
                a = Int.(ones(sz));
            end

            child1 = Folder(folder,[],idxs[a .== 1])
            push!(part.folders,child1)
            push!(folder.children,child1)
            if length(unique(a)) == 2
                child2 = Folder(folder,[],idxs[a .== 2])
                push!(part.folders,child2)
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
    for folder in parents.folders
        for i in folder.idxs
            child = Folder(folder,[],[i])
            push!(leaves.folders,child)
            push!(folder.children,child)
        end
    end
    push!(tree.levels,leaves)

    # hacky: need to make sure all of the tree's levels have the correct
    # ground set. i'm not sure what the best design choice here is actually
    for l in tree.levels
        l.ground_set = IntSet(1:n_points)
    end

    return tree
end


function dual_geometry(data,tree; alpha=1,normalize_=true,center_=true,cutoff=2)
    #=
    computes the geometry (in the form of a similarity matrix) of a dataset
    based on a geometry of its features (rows). this can then be used to
    construct a tree on the data points (columns)

    corresponds to partition_dualgeometry in Coifman's code
    =#

    n_features, n_points = size(data)
    depth = length(tree.levels)
    W = zeros(n_points,n_points)
    # don't integrate over root or leaves; their information is basically useless
    for i in 2:depth-1
        similarity = correlation_similarity(data,tree.levels[i],normalize_=normalize_,center_=center_,cutoff=cutoff)
        similarity = similarity + similarity'
        W += 2.0^(-alpha*i)*similarity
        # this maybe not be exactly right, unless the folder divisions are balanced
    end

    return W
end




#
# ssc affinities: deprecated
#

function ssc_similarity(data,part::Partition; lam=1.0,center_=false,cutoff=10,iters=25,diag_=1.0)
    #=
    solve the lasso-ssc problem on each sub-folder and sum similarity
    matrices. always normalize and center

    the main question is how lambda should scale. in the candes
    paper it is shown that the best choice is 1/sqrt(d), d being the
    dimension of the subspace we're looking for. obviously this is a
    bit complicated in the presence of feature spaces of changing
    dimensionality. one obvious solution is to scale lambda as the
    reduction in feature space, in effect assuming that we keep
    about an even proportion of the subspace when we cut the subspace
    down. but this is odd; a random subspace should retain its
    dimension when we cut out coordinates. so another idea is to
    not modify lambda at all.

    the solves should be via fast projected gradient method, to
    allow for the zero diagonal constraint

    after doing the SSC to get a coefficient matrix C, we derive a
    similarity matrix W as eye(n) + (|C| + |C'|)/2. typically this
    is positive semidefinite.
    =#

    n_features, n_points = size(data)
    if n_features != length(part.ground_set)
        error("partition must be of the rows of data")
    end

    W = zeros(n_points,n_points)
    total_len = 0
    for folder in part.folders
        idxs = collect(folder.idxs)
        if length(idxs) <= cutoff
            continue
        end
        total_len += length(idxs)
        sub_data = data[idxs,:]

        if center_
            sub_data = sub_data .- mean(sub_data,1)
        end
        sub_data = sub_data ./ sqrt.(sum(abs2,sub_data,1))

        # dense matrix arithmetic; maybe sparsify later
        # with a nearest neighbor search
        C = ssc(data,lam;iters=iters)
        W += diag_*eye(n_points) + (abs.(C)+abs.(C'))/2
    end

    if total_len > 0
        W ./= total_len
    end

    return W
end


function soft_threshold(A,lam)
    return sign.(A).*max.(abs.(A)-lam,0)
end


function ssc(data,lam;iters=100)
    #=
    solves min 1/2*||data - data*X||_F^2 + lam*||X||_1 s.t. diag(X) = 0
    using the fast gradient method
    =#

    ATA = data'*data
    n = size(data,2)
    S = zeros(n,n)
    L = eigs(ATA,nev=1,which=:LM)[1][1]
    t = 1.0
    for i in 1:iters
        G = ATA*(S - eye(n))
        S_proj = soft_threshold(S - 1/L*G,lam/L)
        for i in 1:n; S_proj[i,i] = 0; end
        t0 = t
        t = (1 + sqrt(1 + 4*t^2))/2
        S = S_proj + ((t0-1)/t) * (S_proj-S)
    end
    return S
end


function dual_geometry_ssc(data,tree; alpha=1,center_=false,cutoff=10,lam=1,iters=25,diag_=1.0)
    n_features, n_points = size(data)
    depth = length(tree.levels)
    W = zeros(n_points,n_points)
    # don't integrate over root or leaves; their information is basically useless
    for i in 2:depth-1
        similarity = ssc_similarity(data,tree.levels[i],lam=lam,iters=iters,center_=center_,cutoff=cutoff,diag_=diag_)
        W += 2.0^(-alpha*i)*similarity
        # this maybe not be exactly right, unless the folder divisions are balanced
    end

    return W
end


#=

definining a partition tree type --- a hiersrchy of folders
for now, each folder in a partition tree only contains indices,
    not points or other values. it's assume that if the user
    wants to access this information they can do so externally;
    the tree is only here for the tree structure

=#

struct Folder
    parent::Nullable{Folder}  # folder or nothing
    children::Array{Folder,1}
    idxs::IntSet
end

function Folder()
    return Folder(nothing,[],[])
end

function Folder(idxs)
    return Folder(nothing,[],IntSet(idxs))
end


mutable struct PartitionTree
    #=
    root has as its index set the entire found set, and the leaves
    have all singletons
    =#
    ground_set::IntSet
    root::Folder
    leaves::Array{Folder,1}
    levels::Array{Array{Folder,1},1}
end

function PartitionTree(ground_set)
    root = Folder(ground_set)
    tree = PartitionTree(IntSet(ground_set),root,[],[[root],[]])
    for i in ground_set
        leaf = Folder(root,[],IntSet([i]))
        leaf.parent = root
        push!(root.children,leaf)
        push!(tree.leaves,leaf)
        push!(tree.levels[2],leaf)
    end
    return tree
end

function correlation(f::Array,g::Array,tree::PartitionTree; normalize=true,center=true)
    #=
    computes the correlations of f and g over the entire
    partition tree
    =#
end


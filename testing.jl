
# testing all the functions and classes in the project

#
# Folder struct
# 
f = Folder()
f = Folder([1 2 3 4])

#
# Partition struct
#
p = Partition([Folder([1 2 3 4]),Folder([5,6,7,8])])

#
# PartitionTree struct
#
t = PartitionTree([1 2 3 4 5 6 7 8])
assert(t.levels[1].folders[1].inds == IntSet([1 2 3 4 5 6 7 8]))
root = t.levels[1].folders[1]
p.folders[1].parent = root
p.folders[2].parent = root
push!(root.children,p.folders[1])
push!(root.children,p.folders[2])

#
# kmeans_rep function
#
data = zeros(10,100)
data[:,1:50] = randn(10,50)
data[:,51:100] = randn(10,50) + 2
clustering = kmeans_rep(data,2,replications=20)
assert((all(clustering[1:50] .== 2) && all(clustering[51:end] .== 1)) ||
    (all(clustering[51:end] .== 2) && all(clustering[1:50] .== 1)))

#
# correlation_similarity function
#
t = PartitionTree(1:4)
root = t.levels[1].folders[1]

f1 = Folder([1 2]); f2 = Folder([3 4])
f1.parent = f2.parent = root
push!(root.children,f1); push!(root.children,f2)
p1 = Partition([f1,f2])
push!(t.levels,p1)

fs = [Folder([i]) for i in 1:4]
fs[1].parent = fs[2].parent = f1
fs[3].parent = fs[4].parent = f2
push!(f1.children,fs[1])
push!(f1.children,fs[2])
push!(f2.children,fs[3])
push!(f2.children,fs[4])
p2 = Partition(fs)
push!(t.levels,p2)

data = randn(4,4)

W = correlation_similarity(data,p1)

#
# dyadic_tree function
#
vecs = zeros(100,16)
vals = ones(16)
vecs[1:25,:] = randn(25,16)
vecs[26:50,:] = randn(25,16)+1.5
vecs[51:75,:] = randn(25,16)+5
vecs[76:end,:] = randn(25,16)+6.5
t = dyadic_tree(vecs,vals;max_levels=5,n_replicates=20)

#
# dual_geometry
#

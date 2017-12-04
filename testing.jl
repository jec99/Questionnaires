
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


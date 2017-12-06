

include("trees.jl");
include("ncut.jl");

# assume that data is an n x m matrix loaded into memory
# we say that the columns are points and the rows are sensors

# basic
#
data = zeros(200,100);
U1 = qr(randn(200,10))[1];
U2 = qr(randn(200,10))[1];
data[:,1:50] = U1*randn(10,50);
data[:,51:100] = U2*randn(10,50);
perm1 = randperm(200);
perm2 = randperm(100);
data = data[perm1,perm2];
in_1 = sort(sortperm(perm2)[1:50]);
in_2 = sort(sortperm(perm2)[51:100]);


# subspace clustering with irrelevant features
#
n = 1000;
d = 256;
n_subspaces = 10;
n_vecs = Int(n / n_subspaces);
dim_subspace = 32;
irrelevant = 32;
relevant = d - irrelevant;

data = randn(d,n)/sqrt(relevant);
for i in 1:n_subspaces
    U = randn(relevant,dim_subspace)/sqrt(relevant);
    U = qr(U)[1];
    cfs = randn(dim_subspace,n_vecs)/sqrt(dim_subspace);
    data[1:relevant,(i-1)*n_vecs+1:i*n_vecs] = U*cfs;
end

perm1 = randperm(d);
perm2 = randperm(n);
data = data[perm1,perm2];
inverse_perm2 = sortperm(perm2);
membership = [];
sub_size = Int(n/n_subspaces);
for i in 1:n_subspaces
    push!(membership,sort(inverse_perm2[(i-1)*sub_size+1:i*sub_size]))
end


# subspace clustering with irrelevant features, where the
# features depend on the subspace
#
n = 500;
d = 500;
n_subspaces = 5;
n_vecs = Int(n / n_subspaces);
dim_subspace = 20;
relevant = 100;

data = randn(d,n)/sqrt(relevant);
for i in 1:n_subspaces
    U = randn(relevant,dim_subspace)/sqrt(relevant);
    U = qr(U)[1];
    cfs = randn(dim_subspace,n_vecs)/sqrt(dim_subspace);
    strt = rand(0:d-relevant-1);
    data[strt+1:strt+relevant,(i-1)*n_vecs+1:i*n_vecs] = U*cfs;
end

perm1 = randperm(d);
perm2 = randperm(n);
data = data[perm1,perm2];
inverse_perm2 = sortperm(perm2);
membership = [];
for i in 1:n_subspaces
    push!(membership,sort(inverse_perm2[(i-1)*n_vecs+1:i*n_vecs]))
end


#
# settings
#
n_levels = 12;
n_eigs_point = 8;
n_eigs_sensor = 8;
alpha = 0.25;
center_ = true;
iterations = 8;
cutoff = 4;

n_sensors,n_points = size(data);

#
# 1. construct preliminary geometry on points; equivalent to trivial sensor tree
#

W_point = correlation_similarity(data,Partition(1:n_sensors),center_=center_);
vals_point,vecs_point = ncut(W_point,n_eigs_point);
points_tree = dyadic_tree(vecs_point[:,2:end],vals_point[2:end],max_levels=n_levels);

#
# 2. compute initial sensor geometry
#
W_sensor = correlation_similarity(data',Partition(1:n_points),center_=center_);
vals_sensor,vecs_sensor = ncut(W_sensor,n_eigs_sensor);
sensors_tree = dyadic_tree(vecs_sensor[:,2:end],vals_sensor[2:end],max_levels=n_levels);

#
# 3. alternating loop
#
for itr in 1:iterations
    W_point = dual_geometry(data,sensors_tree,alpha=alpha,center_=center_,cutoff=cutoff);
    vals_point,vecs_point = ncut(W_point,n_eigs_point);
    points_tree = dyadic_tree(vecs_point[:,2:end],vals_point[2:end],max_levels=n_levels);

    W_sensor = dual_geometry(data',points_tree,alpha=alpha,center_=center_,cutoff=cutoff);
    vals_sensor,vecs_sensor = ncut(W_sensor,n_eigs_sensor);
    sensors_tree = dyadic_tree(vecs_sensor[:,2:end],vals_sensor[2:end],max_levels=n_levels);
end




# ssc-based similarity metric
#
n_levels = 6;
n_eigs_point = 8;
n_eigs_sensor = 8;
alpha = 0.5;
center_ = false;
iterations = 8;
cutoff = 8;
iters = 25;
lam = 0.01;
diag_ = 0.3;


n_sensors,n_points = size(data);

#
# 1. construct preliminary geometry on points; equivalent to trivial sensor tree
#

W_point = ssc_similarity(data,Partition(1:n_sensors),center_=center_,iters=iters,lam=lam,cutoff=cutoff,diag_=diag_);
vals_point,vecs_point = ncut(W_point,n_eigs_point);
points_tree = dyadic_tree(vecs_point[:,2:end],vals_point[2:end],max_levels=n_levels);

#
# 2. compute initial sensor geometry
#
W_sensor = ssc_similarity(data',Partition(1:n_points),center_=center_,iters=iters,lam=lam,cutoff=cutoff,diag_=diag_);
vals_sensor,vecs_sensor = ncut(W_sensor,n_eigs_sensor);
sensors_tree = dyadic_tree(vecs_sensor[:,2:end],vals_sensor[2:end],max_levels=n_levels);

#
# 3. alternating loop
#
for itr in 1:iterations
    print("\n");print(itr);print("\n\n");
    W_point = dual_geometry_ssc(data,sensors_tree,alpha=alpha,center_=center_,iters=iters,lam=lam,cutoff=cutoff,diag_=diag_);
    vals_point,vecs_point = ncut(W_point,n_eigs_point);
    points_tree = dyadic_tree(vecs_point[:,2:end],vals_point[2:end],max_levels=n_levels);

    W_sensor = dual_geometry_ssc(data',points_tree,alpha=alpha,center_=center_,iters=iters,lam=lam,cutoff=cutoff,diag_=diag_);
    vals_sensor,vecs_sensor = ncut(W_sensor,n_eigs_sensor);
    sensors_tree = dyadic_tree(vecs_sensor[:,2:end],vals_sensor[2:end],max_levels=n_levels);
end

#
# works, is just annoying; separates off folders one at a time, so it takes forever to
# get the true clusters. it's also absurdly slow. the inner product one is better in
# every way.
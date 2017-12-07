

include("trees.jl");
include("ncut.jl");

# assume that data is an n x m matrix loaded into memory
# we say that the columns are points and the rows are sensors


function two_block(d,n,k)
    data = zeros(d,2n)
    U1 = qr(randn(d,k))[1];
    U2 = qr(randn(d,k))[1];
    data[:,1:n] = U1*randn(k,n);
    data[:,n+1:2n] = U2*randn(k,n);
    return data
end


function irrelevant_features(d,n,n_subspaces,dim_subspace,irrelevant)
    n_vecs = Int(n / n_subspaces);
    relevant = d - irrelevant;
    data = randn(d,n)/sqrt(relevant);
    for i in 1:n_subspaces
        U = randn(relevant,dim_subspace)/sqrt(relevant);
        U = qr(U)[1];
        cfs = randn(dim_subspace,n_vecs)/sqrt(dim_subspace);
        data[1:relevant,(i-1)*n_vecs+1:i*n_vecs] = U*cfs;
    end
    return data
end


function irrelevant_features_per_subspace(d,n,n_subspaces,dim_subspace,relevant)
    n_vecs = Int(n / n_subspaces);

    data = randn(d,n)/sqrt(relevant);
    for i in 1:n_subspaces
        U = randn(relevant,dim_subspace)/sqrt(relevant);
        U = qr(U)[1];
        cfs = randn(dim_subspace,n_vecs)/sqrt(dim_subspace);
        strt = rand(0:d-relevant-1);
        data[strt+1:strt+relevant,(i-1)*n_vecs+1:i*n_vecs] = U*cfs;
    end
    return data
end


function organize(data,n_levels,iterations; cutoff=4,n_eigs_point=8,n_eigs_sensor=8,alpha=1,center_=true)
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

    return points_tree, vecs_point, sensors_tree, vecs_sensor
end








#=

# ssc-based similarity metric; doesn't really work
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

=#

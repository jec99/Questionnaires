

#=

regularized top eigensolve function

=#

function ncut(W,n_eigs; normalize_=true,regularizer=eps())
    #=
    finds the top n_eigs eigenvalues/eigenvectors of a symmetric
    matrix, with regularization.

    basic: no sparsity, no randomization
    =#

    offset = 5e-1  # consider dropping
    # should really only be used on symetric matrices, but let's
    # symmetrize "just in case"
    W = (W+W')/2
    n = size(W,1)
    n_eigs = min(n_eigs,n)

    deg = sum(abs.(W),2)  # 'degree' --- sum of rows
    deg_r = (deg .- sum(W,2))/2 + offset
    deg = deg + 2*offset
    deg = deg[:,1]
    deg_r = deg_r[:,1]
    W = W + diagm(deg_r)
    d_invsqrt = 1./sqrt.(deg+regularizer)

    P = (W.*d_invsqrt).*(d_invsqrt')  # multiply on both sides by diagonal

    #=
    vals,vecs = eig(Symmetric(P),n-n_eigs+1:n)  # cols of V are eigenvectors
    # vals,vecs = eigs(Symmetric(P),nev=n_eigs,which=:LM)
    # eigenvectors,eigenvalues ordered increasing; want decreasing
    vecs = vecs[:,end:-1:1]
    vals = vals[end:-1:1]
    =#

    vals,vecs = eigs(Symmetric(P),nev=n_eigs)

    if normalize_
        vecs = vecs.*d_invsqrt
        for i in 1:n_eigs
            vecs[:,i] = vecs[:,i] / norm(vecs[:,i]) * norm(ones(n))
            vecs[:,i] = vecs[:,i] * sign(vecs[1,i])
        end
    end

    return vals,vecs
end


function eigs_fast(A,k; q=2)
    #=
    using block Krylov iteration (Musco & Musco NIPS 2015)

    assuming symmetric A
    =#

    n = size(A,1)
    B = randn(n,k)
    K = zeros(n,q*k)
    K[:,1:k] = A*B
    for i in k:k:k*q-1
        K[:,i+1:i+k] = A*(K[:,i-k+1:i])
    end
    O = qr(K)[1]
    M = O'*(A*O)
    vals,vecs = eigs(Symmetric(M),nev=k,which=:LM)
    return vals,O*vecs
end


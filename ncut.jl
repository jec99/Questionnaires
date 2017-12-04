

#=

regularized top eigensolve function

=#

function ncut(W,n_eigs;normalize_=true,regularizer=eps())
    #=
    finds the top n_eigs eigenvalues/eigenvectors of a symmetric
    matrix, with regularization.

    basic: no sparsity, no randomization
    =#

    offset = 5e-1
    # should really only be used on symetric matrices, but let's
    # symmetrize "just in case"
    W = (W+W')/2
    n = size(W,1)
    n_eigs = min(n_eigs,n)

    deg = sum(abs.(W),2)  # 'degree' --- sum of rows
    deg_r = (d - sum(W,2))/2 + offset
    d = d + 2*offset
    W = W + diagm(dr)
    d_invsqrt = 1/sqrt.(d+regularizer)

    P = (W.*d_invsqrt).*d_invsqrt'  # multiply on both sides by diagonal
    vals,vecs = eig(Symmetric(P),n-n_eigs+1:n)  # cols of V are eigenvectors

    if normalize_
        vecs = vecs.*d_invsqrt
        for i in 1:n_eigs
            vecs[:,i] = vecs[:,i] / norm(vecs[:,i]) * norm(ones(n))
            vecs[:,i] = vecs[:,i] * sign(vecs[1,i])
        end
    end

    return vecs
end


function ncut_fast(W,n_eigs;normalize_=true,regularizer=eps())
    #=
    using simultaneous power iteration and sketching
    and perhaps sparsification
    =#
end
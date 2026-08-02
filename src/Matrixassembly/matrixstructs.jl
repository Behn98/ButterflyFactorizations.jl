"""
    PetrovGalerkinBF{T, ...} <: LinearMaps.LinearMap{T}

The global linear operator representing the Butterfly-factorized boundary element matrix.
Internally, it manages a `BlockSparseMatrix` for near-field interactions and a flat-array
`ButterflyFactorization` structure for the far-field blocks.

It automatically allocates memory-safe thread workspaces to enable zero-allocation,
highly parallel matrix-vector products.
"""
struct PetrovGalerkinBF{
    T,NearInteractionsType,LType<:AbstractMatrix{Int},BFType,treetype
} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    tree::treetype
    BFs::Vector{BFType}
    near_lookup::LType
    far_lookup::LType
    y_thread_buffers::Vector{Vector{T}}
    thread_workspaces::Vector{ThreadButterflyWorkspace{T}}

    function PetrovGalerkinBF{T}(
        nearinteractions, tree, BFs, dim, near_lookup, far_lookup
    ) where {T}
        n_threads = if isdefined(Threads, :maxthreadid)
            Threads.maxthreadid()
        else
            Threads.nthreads() + 1
        end

        # 1. Initialize empty workspaces for each thread (max depth 20 levels)
        thread_ws = [ThreadButterflyWorkspace{T}(20) for _ in 1:n_threads]

        # 2. Initialize thread-local output buffers
        thread_y = [zeros(T, dim[1]) for _ in 1:n_threads]

        return new{T,typeof(nearinteractions),typeof(near_lookup),eltype(BFs),typeof(tree)}(
            nearinteractions, dim, tree, BFs, near_lookup, far_lookup, thread_y, thread_ws
        )
    end
end

"""
    farmatrix(mat::PetrovGalerkinBF)

Extracts and isolates the far-field operator by zeroing out the near-field interactions.
Useful for debugging operator splits and analyzing compression error.
"""
function farmatrix(
    mat::PetrovGalerkinBF{T}; scheduler=OhMyThreads.SerialScheduler()
) where {T}
    return PetrovGalerkinBF{T}(
        BlockSparseMatrix(Matrix{T}[], Int[], Int[], mat.dim; scheduler=scheduler),
        mat.tree,
        mat.BFs,
        mat.dim,
        mat.near_lookup,
        mat.far_lookup,
    )
end

"""
    PetrovGalerkinBF_Mat{T, ...} <: LinearMaps.LinearMap{T}

The legacy/sparse-matrix format for the global Butterfly factorized operator.
Uses explicit sparse block-diagonal matrices for the far-field interactions.
"""
struct PetrovGalerkinBF_Mat{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    farinteractions::Vector{Tuple{Int,Int}}
    BFs::Vector{ButterflyFactorization_Mat{T}}

    function PetrovGalerkinBF_Mat{T}(nearinteractions, farinteractions, BFs, dim) where {T}
        return new{T,typeof(nearinteractions)}(nearinteractions, dim, farinteractions, BFs)
    end
end

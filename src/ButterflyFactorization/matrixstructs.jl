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
    thread_workspaces::Vector{ThreadButterflyWorkspace{T}} # 🚀 Only n_threads workspaces!

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

function farmatrix(
    mat::PetrovGalerkinBF{T}; scheduler=OhMyThreads.SerialScheduler()
) where {T}
    return PetrovGalerkinBF{T}(
        BlockSparseMatrix(Matrix{ComplexF64}[], Int[], Int[], mat.dim; scheduler=scheduler),
        mat.tree,
        mat.BFs,
        mat.dim,
        mat.near_lookup,
        mat.far_lookup,
    )
end

struct PetrovGalerkinBF_Mat{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    #tree::H2Trees.BlockTree
    farinteractions::Vector{Tuple{Int,Int}}           #observernodeid --> sourcenodeid
    BFs::Vector{ButterflyFactorization_Mat}
    function PetrovGalerkinBF_Mat{T}(
        nearinteractions,
        #tree,
        farinteractions,
        BFs,
        dim,
    ) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            dim,
            # Here come all other fields needed for the ButterflyFactorizations
            #tree,#::H2Trees.BlockTree
            farinteractions,      #observernodeid --> sourcenodeid
            BFs,#::Vector{BF}
        )
    end
end

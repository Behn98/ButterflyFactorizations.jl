struct PetrovGalerkinBF{T,NearInteractionsType,LType<:AbstractMatrix{Int},BFType,WSType} <:
       LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree
    BFs::Vector{BFType}
    workspaces::Vector{WSType}   # 🚀 Added to hold pre-allocated workspaces
    near_lookup::LType
    far_lookup::LType

    function PetrovGalerkinBF{T}(
        nearinteractions, tree, BFs, workspaces, dim, near_lookup, far_lookup
    ) where {T}
        return new{
            T,typeof(nearinteractions),typeof(near_lookup),eltype(BFs),eltype(workspaces)
        }(
            nearinteractions, dim, tree, BFs, workspaces, near_lookup, far_lookup
        )
    end
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

abstract type AbstractBlockView{T} <: LinearMaps.LinearMap{T} end

# Fixed the syntax typo here (<: instead of < ref)
struct NearBlockView{T,M<:AbstractMatrix{T}} <: AbstractBlockView{T}
    obs_id::Int
    src_id::Int
    dim::Tuple{Int,Int}
    matrix::M
end

struct FarBlockView{T,BFType} <: AbstractBlockView{T}
    obs_id::Int
    src_id::Int
    dim::Tuple{Int,Int}
    bf::BFType
end

# For parent/composite nodes that aren't leaves or direct BFs
struct CompositeBlockView{
    T,NearInteractionsType,LType<:SparseArrays.SparseMatrixCSC{Int,Int}
} <: AbstractBlockView{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    BFs::Vector{ButterflyFactorization{T}}
    near_lookup::LType
    far_lookup::LType
    function CompositeBlockView{T}(
        nearinteractions, dim, BFs, near_lookup, far_lookup
    ) where {T}
        return new{T,typeof(nearinteractions),typeof(near_lookup)}(
            nearinteractions, dim, BFs, near_lookup, far_lookup
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

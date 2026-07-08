struct PetrovGalerkinBF{
    T,NearInteractionsType,LType<:SparseArrays.SparseMatrixCSC{Int,Int}
} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree
    BFs::Vector{BF}
    near_lookup::LType
    far_lookup::LType

    function PetrovGalerkinBF{T}(
        nearinteractions, tree, BFs, dim, near_lookup, far_lookup
    ) where {T}
        return new{T,typeof(nearinteractions),typeof(near_lookup)}(
            nearinteractions, dim, tree, BFs, near_lookup, far_lookup
        )
    end
end

struct FlatPGBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree
    BFs::Vector{FlatBF}
    function FlatPGBF{T}(nearinteractions, dim, tree, BFs) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            dim,
            tree,#::H2Trees.BlockTree
            BFs,#::Vector{FlatBF}
        )
    end
end

struct PetrovGalerkinBF_mats{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    #tree::H2Trees.BlockTree
    farinteractions::Vector{Tuple{Int,Int}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF_Mats}
    function PetrovGalerkinBF_mats{T}(
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
    BFs::Vector{BF}
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

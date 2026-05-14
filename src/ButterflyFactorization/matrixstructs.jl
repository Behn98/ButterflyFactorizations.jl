struct PetrovGalerkinBF_mats{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    #tree::H2Trees.BlockTree
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
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

struct PetrovGalerkinBF{T,NearInteractionsType} <: LinearMaps.LinearMap{T}
    nearinteractions::NearInteractionsType
    dim::Tuple{Int,Int}
    tree::H2Trees.BlockTree
    farinteractions::Dict{Int64,Vector{Int64}}           #observernodeid --> sourcenodeid
    BFs::Vector{BF}
    function PetrovGalerkinBF{T}(
        nearinteractions, tree, farinteractions, BFs, dim
    ) where {T}
        return new{T,typeof(nearinteractions)}(
            nearinteractions,
            dim,
            tree,#::H2Trees.BlockTree
            farinteractions,           #observernodeid --> sourcenodeid
            BFs,#::Vector{BF}
        )
    end
end

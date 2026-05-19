@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorizations.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y .+= A.nearinteractions * x
    i = 1
    for (NO, source_nodes) in A.farinteractions
        for NS in source_nodes
            gs = H2Trees.values(A.tree.trialcluster, A.BFs[i].NS)
            go = H2Trees.values(A.tree.testcluster, A.BFs[i].NO)
            y[go] .+= apply_BF(A.BFs[i], x[gs])
            i += 1
        end
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        y[gs] .+= apply_BF(transpose(At.lmap.BFs[i]), x[go])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        y[gs] .+= apply_BF(At.lmap.BFs[i]', x[go])
        # OBS: Använd din apply_BF_adjoint funktion här!
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF_mats,
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y .+= A.nearinteractions * x

    for i in eachindex(A.BFs)
        y[A.BFs[i].PermP] .+= applyBF_Mats(A.BFs[i], x[A.BFs[i].PermQ])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyBF_Mats(
            transpose(At.lmap.BFs[i]), x[At.lmap.BFs[i].PermP]
        )
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyBF_Mats(At.lmap.BFs[i]', x[At.lmap.BFs[i].PermP])
    end
    return y
end

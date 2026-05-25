@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorizations.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y_near = A.nearinteractions * x
    y .+= y_near
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()

    @tasks for i in eachindex(A.BFs)
        @set scheduler = DynamicScheduler()

        bf = A.BFs[i]
        gs = H2Trees.values(A.tree.trialcluster, bf.NS)
        go = H2Trees.values(A.tree.testcluster, bf.NO)

        # Beräkna resultatet för blocket (detta sker helt parallellt)
        res = apply_BF(bf, x[gs]; scheduler=OhMyThreads.SerialScheduler())

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[go] .+= res
        end
    end
    BLAS.set_num_threads(old_blas)

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
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()

    @tasks for i in eachindex(At.lmap.BFs)
        @set scheduler = DynamicScheduler()
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        # Beräkna resultatet för blocket (detta sker helt parallellt)
        res = apply_BF(
            transpose(At.lmap.BFs[i]), x[go]; scheduler=OhMyThreads.SerialScheduler()
        )

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[gs] .+= res
        end
    end
    # Återställ BLAS
    BLAS.set_num_threads(old_blas)
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
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()
    @tasks for i in eachindex(At.lmap.BFs)
        @set scheduler = DynamicScheduler()
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        res = apply_BF(At.lmap.BFs[i]', x[go]; scheduler=OhMyThreads.SerialScheduler())

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[gs] .+= res
        end
    end
    BLAS.set_num_threads(old_blas)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorizations.FlatPGBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y_near = A.nearinteractions * x
    y .+= y_near
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()

    @tasks for i in eachindex(A.BFs)
        @set scheduler = DynamicScheduler()

        bf = A.BFs[i]
        gs = H2Trees.values(A.tree.trialcluster, bf.NS)
        go = H2Trees.values(A.tree.testcluster, bf.NO)

        # Beräkna resultatet för blocket (detta sker helt parallellt)
        res = mul_flat_bf(bf, x[gs]; scheduler=OhMyThreads.SerialScheduler())

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[go] .+= res
        end
    end
    # Återställ BLAS
    BLAS.set_num_threads(old_blas)

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.FlatPGBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()

    @tasks for i in eachindex(At.lmap.BFs)
        @set scheduler = DynamicScheduler()
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        # Beräkna resultatet för blocket (detta sker helt parallellt)
        res = mul_flat_bf(
            transpose(At.lmap.BFs[i]), x[go]; scheduler=OhMyThreads.SerialScheduler()
        )

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[gs] .+= res
        end
    end
    BLAS.set_num_threads(old_blas)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.FlatPGBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    y_lock = Threads.SpinLock()

    @tasks for i in eachindex(At.lmap.BFs)
        @set scheduler = DynamicScheduler()
        gs = H2Trees.values(At.lmap.tree.trialcluster, At.lmap.BFs[i].NS)
        go = H2Trees.values(At.lmap.tree.testcluster, At.lmap.BFs[i].NO)

        # Beräkna resultatet för blocket (detta sker helt parallellt)
        res = mul_flat_bf(At.lmap.BFs[i]', x[go]; scheduler=OhMyThreads.SerialScheduler())

        # Lås kortvarigt när vi uppdaterar 'y' så att inte trådar skriver över varandra
        lock(y_lock) do
            y[gs] .+= res
        end
    end
    BLAS.set_num_threads(old_blas)
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

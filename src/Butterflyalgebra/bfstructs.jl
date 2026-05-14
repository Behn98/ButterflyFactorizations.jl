struct AlgBF
    dim::Tuple{Int,Int}
    Q::Dict{Int,Matrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,Matrix{ComplexF64}}
end

struct BF
    Q::Dict{Int,AbstractMatrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,AbstractMatrix{ComplexF64}}
    PermQ::Dict{Int,Vector{Int}}
    PermP::Dict{Int,Vector{Int}}
    dim::Tuple{Int,Int}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    BF(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ) = new(
        Q, R, P, PermQ, PermP, dim, NS, NO, k, τ
    )
end

struct BF_Mats
    Q::AbstractMatrix{ComplexF64}
    R::Vector{AbstractMatrix{ComplexF64}}
    P::AbstractMatrix{ComplexF64}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    PermP::Vector{Int}
    PermQ::Vector{Int}
    BF_Mats(Q, R, P, NS, NO, k, τ, PermP, PermQ) = new(Q, R, P, NS, NO, k, τ, PermP, PermQ)
end
#AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices

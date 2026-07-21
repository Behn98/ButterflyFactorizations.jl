"""
    ButterflyFactorizations

A Julia module for constructing, manipulating, and applying Butterfly Factorizations.

This package provides hierarchical low-rank approximation techniques typically used for
highly oscillatory integral operators. It leverages tree-based domain partitioning and
includes functionality for:

  - Structured matrix-vector and matrix-matrix products.
  - Algebraic operations on butterfly structures (addition, multiplication, and
    recompression).
  - Petrov-Galerkin block matrix assembly of kernel matrices, with extension support for
    boundary element frameworks.

# Main API

**Compressors:**

  - [`PartialQR`](@ref): QR-based low-rank approximation

**Kernel matrices:**

  - [`BEASTKernelMatrix`](@ref): BEAST boundary integral operator matrices

# Example

check the test folder/Readme.md for example usage of the API, including how to construct a
`PetrovGalerkinBF` from a BEAST operator and apply it to a vector.

# See also

  - BEAST.jl for boundary integral operators
  - H2Trees.jl for hierarchical clustering
  - BlockSparseMatrices.jl or Sparse SparseArrays.jl for sparse block storage
"""
module ButterflyFactorizations

using BlockSparseMatrices
using H2Trees
using LinearAlgebra
using LinearMaps
using StaticArrays
using Random
using OhMyThreads
using LowRankApprox
using SparseArrays

export PetrovGalerkinBF, PetrovGalerkinBF_mats, subroutine_BF, subroutine_BF_mats, PartialQR
export mulBFs, add_eqbfs, recompress_BF, apply_BF, FlatBF, flattenmatrix, splitmulbf

#Helper funcitons
include("auxillaries.jl")

#Kernelmatrix import
include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

#Butterfly algebra --> any Block related functions
include("Butterflyalgebra/bfstructs.jl")
include("Butterflyalgebra/bfstructfcts.jl")
include("Butterflyalgebra/flatbf.jl")
include("Butterflyalgebra/bfadjtr.jl")
include("Butterflyalgebra/bfdims.jl")
include("Butterflyalgebra/bfvector.jl")
include("Butterflyalgebra/bfmatrix.jl")
include("Butterflyalgebra/algrecomp.jl")
include("Butterflyalgebra/bfbfadd.jl")
include("Butterflyalgebra/bfbfmul.jl")
include("Butterflyalgebra/bfsplitmul.jl")
include("Butterflyalgebra/bfsplit.jl")

#Tree traversale and Butterfly construction
include("intlists.jl")
include("compressors.jl")
include("subroutines.jl")
include("bfassembly.jl")

#Full Matrix Assembly
include("ButterflyFactorization/matrixstructs.jl")
include("ButterflyFactorization/petrovgalerkinbf.jl")

include("matrixalgebra/dims.jl")
include("matrixalgebra/csrmatrix.jl")
include("matrixalgebra/indexing.jl")
include("matrixalgebra/matrixvector.jl")
include("matrixalgebra/matrixmatrix.jl")

include("showfcts.jl")

end

@meta
CurrentModule = ButterflyFactorizations

# ButterflyFactorizations.jl

ButterflyFactorizations.jl provides a fast algebraic compression framework for high-frequency electromagnetic boundary integral equations. By exploiting the complementary low-rank property of far-field interactions, it reduces the complexity of dense, oscillatory kernel matrices from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log N)$.The package integrates natively with H2Trees.jl and features thread-safe, flat-array workspaces optimized for zero-allocation matrix-vector products in iterative solvers.

## Installation

pkg> add [https://github.com/Behn98/ButterflyFactorizations.jl.git](https://github.com/Behn98/ButterflyFactorizations.jl.git)

## Quick Start

```julia
using BEAST, CompScienceMeshes, H2Trees
using ButterflyFactorizations, LinearAlgebra, OhMyThreads

# Restrict BLAS threads to prevent oversubscription during custom parallelization
LinearAlgebra.BLAS.set_num_threads(1)

# Problem Setup
h = 0.1; k = 2 * pi / (10 * h)
m = meshsphere(1.0, h)
X = raviartthomas(m)
op = Maxwell3D.singlelayer(; wavenumber=k)

# Tree Decomposition & Factorization
tree = ButterflyFactorizations.build_bisection_tree(X.pos; max_points = 100)
blktree = BlockTree(tree, tree)

Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
    op, X, X, blktree, k;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    scheduler=OhMyThreads.DynamicScheduler()
)
```

## API Reference

### Core Types & Operators

```@docs
ButterflyFactorizations
ButterflyFactorization
ButterflyFactorization_Mat
PetrovGalerkinBF
PetrovGalerkinBF_Mat
```

### Assembly & Compressors
```@docs
assemble_BF
assemble_BF_Mat
PartialQR
```

### Workspaces & Internal Structures

```@docs
ButterflyWorkspace
ThreadButterflyWorkspace
ButterflyBlock
ButterflyLevel
```

### Algebra & Recompression

```@docs
mulBFs
add_eqbfs
recompress_BF
```


@meta
CurrentModule = ButterflyFactorizations

# ButterflyFactorizations.jl

Documentation for [ButterflyFactorizations](https://github.com/Behn98/ButterflyFactorizations.jl).

## Core Types & Operators

```@docs
ButterflyFactorizations
ButterflyFactorization
PetrovGalerkinBF
PetrovGalerkinBF_Mat
```

## Assembly & Compressors
```@docs
assemble_BF
assemble_BF_Mat
PartialQR
BEASTKernelMatrix
```

## Workspaces & Memory Management

```@docs
ButterflyWorkspace
ThreadButterflyWorkspace
PartialQRWorkspace
```

## Algebra & Recompression

```@docs
mulBFs
add_eqbfs
recompress_BF
```


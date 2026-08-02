@meta
CurrentModule = ButterflyFactorizations

# ButterflyFactorizations.jl

Documentation for [ButterflyFactorizations](https://github.com/Behn98/ButterflyFactorizations.jl).

## Core Types & Operators

```@docs
ButterflyFactorizations
ButterflyFactorization
ButterflyFactorization_Mat
PetrovGalerkinBF
PetrovGalerkinBF_Mat
```

## Assembly & Compressors
```@docs
assemble_BF
assemble_BF_Mat
PartialQR
```

## Workspaces & Internal Structures

```@docs
ButterflyWorkspace
ThreadButterflyWorkspace
ButterflyBlock
ButterflyLevel
```

## Algebra & Recompression

```@docs
mulBFs
add_eqbfs
recompress_BF
```


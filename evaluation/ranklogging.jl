using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using OhMyThreads
using ButterflyFactorizations
using LinearAlgebra
using BenchmarkTools
using Printf
using PlotlyJS
using Random
using SparseArrays
using Statistics

function fit_rank_parameters(logger::ButterflyFactorizations.RankLogger; safety_margin=2.0)
    all_records = reduce(vcat, logger.buffers)
    if isempty(all_records)
        error("No rank records found in logger.")
    end

    N_samples = length(all_records)
    X = zeros(Float64, N_samples, 2)
    y = zeros(Float64, N_samples)
    x2_vec = zeros(Float64, N_samples)

    for i in 1:N_samples
        X[i, 1] = all_records[i][1] # x1 (Geometric Term)
        X[i, 2] = all_records[i][2] # x2 (log(1/ε))
        x2_vec[i] = all_records[i][2]
        y[i] = all_records[i][3]    # actual computed rank 'r'
    end

    # 1. Fit Geometric Estimator (C, Cε) via Least Squares
    beta = X \ y
    C_fit = max(0.1, beta[1] * safety_margin)
    Cε_fit = max(0.5, beta[2] * safety_margin)

    # 2. Fit Butterfly Estimator (Cτ)
    # We use the absolute maximum observed ratio to be ULTRA-SAFE.
    # We then multiply by 1.25 to ensure the initial sample size is always > 125%
    # of the true rank, mathematically guaranteeing the 80% adaptive trigger never fires.
    max_ratio = maximum(y ./ x2_vec)
    Cτ_fit = max(0.5, max_ratio * 1.25)

    println("=== Rank Estimator Calibration Results ===")
    println("Sample size    : $N_samples blocks")
    println("--- Geometric Estimator (FMM / Standard) ---")
    println(
        "Fitted C       : $(round(beta[1], digits=4))  --> Tuned: $(round(C_fit, digits=4))"
    )
    println(
        "Fitted Cε      : $(round(beta[2], digits=4))  --> Tuned: $(round(Cε_fit, digits=4))",
    )
    println("--- Butterfly Estimator (High-Frequency) ---")
    println(
        "Max Ratio      : $(round(max_ratio, digits=4))  --> Tuned (1.25x): $(round(Cτ_fit, digits=4))",
    )

    return C_fit, Cε_fit, Cτ_fit
end

# 🚀 Upgraded CSV export function to log BOTH estimators for your TikZ plots!
function export_rank_data_to_csv(
    logger::ButterflyFactorizations.RankLogger,
    C_fit::Float64,
    Cε_fit::Float64,
    Cτ_fit::Float64,
    filename::String,
)
    open(filename, "w") do io
        # Write CSV Header
        write(io, "x1,x2,actual_rank,est_geom,est_bf,err_geom,err_bf\n")

        # Flatten and write thread-local buffers
        for buf in logger.buffers
            for (x1, x2, r_actual) in buf
                r_geom = C_fit * x1 + Cε_fit * x2
                r_bf = Cτ_fit * x2

                err_geom = r_geom - r_actual
                err_bf = r_bf - r_actual

                write(io, "$x1,$x2,$r_actual,$r_geom,$r_bf,$err_geom,$err_bf\n")
            end
        end
    end
    return println("Saved raw rank data to $filename for TikZ plotting.")
end

function plot_rank_diagnostics(
    logger::ButterflyFactorizations.RankLogger,
    C_fit::Float64,
    Cε_fit::Float64,
    Cτ_fit::Float64,
    title_suffix::String,
)
    x1_vals = Float64[]
    x2_vals = Float64[]
    actual_ranks = Float64[]
    est_ranks_geom = Float64[]
    est_ranks_bf = Float64[]

    for buf in logger.buffers
        for (x1, x2, r_actual) in buf
            push!(x1_vals, x1)
            push!(x2_vals, x2)
            push!(actual_ranks, r_actual)

            push!(est_ranks_geom, C_fit * x1 + Cε_fit * x2)
            push!(est_ranks_bf, Cτ_fit * x2)
        end
    end

    if isempty(actual_ranks)
        error("No data found in logger.")
    end

    fig = make_subplots(;
        rows=2,
        cols=2,
        subplot_titles=[
            "1. Guestimate vs. Actual Rank" "2. Rank vs. Geometric Factor (x1)";
            "3. Estimation Error (Geom vs BF)" "4. Residuals vs. Estimated Rank (Geom)"
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    max_val = max(maximum(est_ranks_geom), maximum(est_ranks_bf), maximum(actual_ranks))

    # --- Plot 1: Guestimate vs. Actual (Parity Plot) ---
    trace_geom = scatter(;
        x=est_ranks_geom,
        y=actual_ranks,
        mode="markers",
        marker=attr(; size=4, color="royalblue", opacity=0.6),
        name="Geometric Est.",
    )
    trace_bf = scatter(;
        x=est_ranks_bf,
        y=actual_ranks,
        mode="markers",
        marker=attr(; size=4, color="coral", opacity=0.6),
        name="Butterfly Est.",
    )
    trace_ideal = scatter(;
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=attr(; color="black", dash="dash"),
        name="Ideal",
    )
    trace_adaptive = scatter(;
        x=[0, max_val],
        y=[0, 0.8 * max_val],
        mode="lines",
        line=attr(; color="red", dash="dot"),
        name="80% Resample Limit",
    )

    add_trace!(fig, trace_geom; row=1, col=1)
    add_trace!(fig, trace_bf; row=1, col=1)
    add_trace!(fig, trace_ideal; row=1, col=1)
    add_trace!(fig, trace_adaptive; row=1, col=1)

    # --- Plot 2: Rank vs Geometric Factor (x1) ---
    trace2 = scatter(;
        x=x1_vals,
        y=actual_ranks,
        mode="markers",
        marker=attr(; size=4, color="purple", opacity=0.5),
        name="Actual Ranks",
    )
    add_trace!(fig, trace2; row=1, col=2)

    # --- Plot 3: Estimation Error Histogram ---
    err_geom = est_ranks_geom .- actual_ranks
    err_bf = est_ranks_bf .- actual_ranks

    trace3_geom = histogram(;
        x=err_geom, nbinsx=40, marker_color="royalblue", name="Geom Error", opacity=0.7
    )
    trace3_bf = histogram(;
        x=err_bf, nbinsx=40, marker_color="coral", name="BF Error", opacity=0.7
    )

    add_trace!(fig, trace3_geom; row=2, col=1)
    add_trace!(fig, trace3_bf; row=2, col=1)

    # --- Plot 4: Residuals vs Estimated (Geometric) ---
    trace4 = scatter(;
        x=est_ranks_geom,
        y=err_geom,
        mode="markers",
        marker=attr(; size=4, color="seagreen", opacity=0.5),
        showlegend=false,
    )
    trace_zero = scatter(;
        x=[0, max_val],
        y=[0, 0],
        mode="lines",
        line=attr(; color="black", dash="dash"),
        showlegend=false,
    )
    add_trace!(fig, trace4; row=2, col=2)
    add_trace!(fig, trace_zero; row=2, col=2)

    # Update Layout
    relayout!(
        fig;
        title_text="Rank Estimator Diagnostics: $title_suffix",
        height=800,
        width=1200,
        template="plotly_white",
        barmode="overlay",
        xaxis_title="Estimated Rank",
        yaxis_title="Actual Rank",
        xaxis2_title="Geometric Factor x1 (Log Scale)",
        yaxis2_title="Actual Rank",
        xaxis2_type="log",
        xaxis3_title="Estimation Error (Est - Actual)",
        yaxis3_title="Count",
        xaxis4_title="Estimated Rank (Geom)",
        yaxis4_title="Error",
    )

    return fig
end

# ==============================================================================
# --- SETUP GEOMETRY & EXECUTION ---
# ==============================================================================
# 🚀 Use a small N for calibration!
h = 0.05
lambda = 10 * h
k = 2 * pi / lambda
op = Maxwell3D.singlelayer(; wavenumber=k)
m = meshsphere(1.0, h)
X = raviartthomas(m)
N = length(X)

BLAS.set_num_threads(1)
tol = 1e-3

tree_types = [:KMeansTree, :BisectionTree, :TwoNTree]

for tree_type in tree_types
    println("\n**************************************************")
    println(" 🌲 Building Tree Architecture: $tree_type")
    println("**************************************************")

    if tree_type == :KMeansTree
        tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
    elseif tree_type == :BisectionTree
        tree = ButterflyFactorizations.build_bisection_tree(X.pos; max_points=100)
    elseif tree_type == :TwoNTree
        tree = H2Trees.TwoNTree(X, h)
    end

    blktree = H2Trees.BlockTree(tree, tree)

    criteria_to_test = [
        (
            :isFarFunctor,
            ButterflyFactorizations.isFarFunctor(
                ButterflyFactorizations.tree_parameters(
                    tree, ButterflyFactorizations.isFarFunctor
                ).α,
            ),
        ),
        (
            :CenterDistanceAdmissibility,
            ButterflyFactorizations.CenterDistanceAdmissibility(
                ButterflyFactorizations.tree_parameters(
                    tree, ButterflyFactorizations.CenterDistanceAdmissibility
                ).β,
            ),
        ),
    ]

    for (name, admissibility_functor) in criteria_to_test
        println("\n==================================================================")
        println(" Starting Rank Calibration for: $tree_type + $name")
        println("==================================================================")

        logger = ButterflyFactorizations.RankLogger()
        compressor = ButterflyFactorizations.PartialQR(logger)

        # Assemble using the new Functor architecture to log the variables
        A = ButterflyFactorizations.PetrovGalerkinBF(
            op,
            X,
            X,
            blktree,
            k;
            compressor=compressor,
            tol=tol,
            admissibility=admissibility_functor,

            # 🚀 NEW: Force the Geometric estimator with neutral weights (1.0) so it logs pure x1 and x2
            rankestimator=ButterflyFactorizations.GeometricRankEstimator(1.0, 1.0; Rmin=3),

            scheduler=OhMyThreads.DynamicScheduler(),
            acctype=ComplexF64,
            minbflvl=2,
            adaptive=true,   # MUST BE TRUE FOR CALIBRATION
            unbalancedints=false,
            leafcomp=true,
            leafimbalance=false,
        )

        # 🚀 Fit all 3 parameters!
        C_opt, Cε_opt, Cτ_opt = fit_rank_parameters(logger; safety_margin=2.0)

        # Plot and save HTML
        fig = plot_rank_diagnostics(logger, C_opt, Cε_opt, Cτ_opt, "$tree_type | $name")
        display(fig)

        html_filename = "plot_rank_calibration_$(tree_type)_$(name).html"
        savefig(fig, html_filename)
        println("Saved HTML diagnostics to $html_filename")

        # 🚀 Dump all raw data to CSV for TikZ
        csv_filename = "rank_calibration_data_$(tree_type)_$(name).csv"
        export_rank_data_to_csv(logger, C_opt, Cε_opt, Cτ_opt, csv_filename)
    end
end

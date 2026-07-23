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

function fit_rank_parameters(logger::ButterflyFactorizations.RankLogger; safety_margin=1.15)
    # Aggregate data from all threads
    all_records = reduce(vcat, logger.buffers)
    if isempty(all_records)
        error("No rank records found in logger.")
    end

    N = length(all_records)
    X = zeros(Float64, N, 2)
    y = zeros(Float64, N)

    for i in 1:N
        X[i, 1] = all_records[i][1] # x1 term
        X[i, 2] = all_records[i][2] # x2 term
        y[i] = all_records[i][3] # actual rank 'r'
    end

    # Solve least squares: X * [C, Cε] ≈ y
    beta = X \ y

    C_fit = max(0.1, beta[1] * safety_margin)
    Cε_fit = max(0.5, beta[2] * safety_margin)

    println("=== Rank Estimator Calibration Results ===")
    println("Sample size    : $N blocks")
    println(
        "Fitted C       : $(round(beta[1], digits=4))  --> Safety tuned ($(safety_margin)x): $(round(C_fit, digits=4))",
    )
    println(
        "Fitted Cε      : $(round(beta[2], digits=4))  --> Safety tuned ($(safety_margin)x): $(round(Cε_fit, digits=4))",
    )
    return C_fit, Cε_fit
end

function plot_rank_diagnostics(
    logger::ButterflyFactorizations.RankLogger, C_fit::Float64, Cε_fit::Float64
)
    # 1. Flatten Thread-Local Buffers
    x1_vals = Float64[]
    x2_vals = Float64[]
    actual_ranks = Float64[]
    est_ranks = Float64[]
    errors = Float64[]

    for buf in logger.buffers
        for (x1, x2, r_actual) in buf
            push!(x1_vals, x1)
            push!(x2_vals, x2)
            push!(actual_ranks, r_actual)

            # Reconstruct the guestimate using your fitted parameters
            r_est = C_fit * x1 + Cε_fit * x2
            push!(est_ranks, r_est)

            # Error > 0: Over-estimated (Safe padding)
            # Error < 0: Under-estimated (Triggered expensive resampling)
            push!(errors, r_est - r_actual)
        end
    end

    if isempty(actual_ranks)
        error("No data found in logger. Did you run the assembly?")
    end

    # 2. Create the Subplot Layout (Using the 2x2 Matrix fix!)
    fig = make_subplots(;
        rows=2,
        cols=2,
        subplot_titles=[
            "1. Guestimate vs. Actual Rank" "2. Rank vs. Geometric Factor (x1)";
            "3. Estimation Error (Safety Margin)" "4. Residuals vs. Estimated Rank"
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    max_val = max(maximum(est_ranks), maximum(actual_ranks))

    # --- Plot 1: Guestimate vs. Actual (Parity Plot) ---
    trace1 = scatter(;
        x=est_ranks,
        y=actual_ranks,
        mode="markers",
        marker=attr(;
            size=4,
            color=x1_vals,
            colorscale="Viridis",
            showscale=true,
            colorbar=attr(; title="x1 (Geom)", x=0.45, len=0.45, y=0.75),
        ),
        name="Blocks",
        hoverinfo="x+y",
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

    add_trace!(fig, trace1; row=1, col=1)
    add_trace!(fig, trace_ideal; row=1, col=1)
    add_trace!(fig, trace_adaptive; row=1, col=1)

    # --- Plot 2: Rank vs Geometric Factor (x1) ---
    # Because x1 spans orders of magnitude, a log-x scale shows the relationship beautifully
    trace2 = scatter(;
        x=x1_vals,
        y=actual_ranks,
        mode="markers",
        marker=attr(; size=4, color="royalblue", opacity=0.5),
        name="Actual Ranks",
    )
    add_trace!(fig, trace2; row=1, col=2)

    # --- Plot 3: Estimation Error Histogram ---
    trace3 = histogram(;
        x=errors, nbinsx=40, marker_color="seagreen", name="Error", histnorm="probability"
    )
    add_trace!(fig, trace3; row=2, col=1)

    # --- Plot 4: Residuals vs Estimated ---
    trace4 = scatter(;
        x=est_ranks,
        y=errors,
        mode="markers",
        marker=attr(; size=4, color="coral", opacity=0.5),
        name="Residuals",
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

    # 3. Update Layout
    relayout!(
        fig;
        title_text="Butterfly Factorization: Rank Estimator Diagnostics",
        height=800,
        width=1200,
        template="plotly_white",
        showlegend=false,
        xaxis_title="Estimated Rank (Guestimate)",
        yaxis_title="Actual Computed Rank",
        xaxis2_title="Geometric Factor x1 (Log Scale)",
        yaxis2_title="Actual Rank",
        xaxis2_type="log", # 🚀 Makes x1 much easier to read!
        xaxis3_title="Estimation Error (Est - Actual)",
        yaxis3_title="Probability Density",
        xaxis4_title="Estimated Rank",
        yaxis4_title="Error (Est - Actual)",
    )

    return fig
end

# How to call it (assuming fit_rank_parameters returned C_fit and Cε_fit):
# C_fit, Cε_fit = fit_rank_parameters(logger, safety_margin=1.15)
# fig = plot_rank_diagnostics(logger, C_fit, Cε_fit)
# display(fig)

h = 0.1
lambda = 10 * h
k = 2 * pi / lambda
op = Maxwell3D.singlelayer(; wavenumber=k)
m = meshsphere(1.0, h)
X = raviartthomas(m)
N = length(X)

# Bygg träd
#tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
#blktree = H2Trees.BlockTree(tree, tree)
Ttree = H2Trees.TwoNTree(X, h;)#minvalues=100
Stree = H2Trees.TwoNTree(X, h;)#minvalues=100
blktree = BlockTree(Ttree, Stree)
#As = assemble(op, X, X)

# Create logger and compressor
logger = ButterflyFactorizations.RankLogger()
compressor = ButterflyFactorizations.PartialQR(logger)
BLAS.set_num_threads(1) # Avoid nested threading issues
# Assemble PetrovGalerkinBF as normal on a representative mesh
A = ButterflyFactorizations.PetrovGalerkinBF(op, X, X, blktree, k; compressor=compressor)

xtest = rand(ComplexF64, size(A, 2))
xs = A * xtest
xs2 = As * xtest
reldif = norm(xs - xs2) / norm(xs2)

# Fit C and Cε
C_opt, Cε_opt = fit_rank_parameters(logger)

plot_rank_diagnostics(logger, C_opt, Cε_opt)

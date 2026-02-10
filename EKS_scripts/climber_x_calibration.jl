using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using JLD2
using Dates
using Printf
using NCDatasets
using Distributions

# Include job management and summary statistics
include("eks_job_management.jl")
include("climber_summary_stats.jl")

# ============================================
# CLIMBER-X CONFIGURATION
# ============================================

const CLIMBER_X_DIR = "/home/karinako/climber-x"
const RUNME_SCRIPT = joinpath(CLIMBER_X_DIR, "runme")
const DEFAULT_RUN_OUTPUT = "/p/tmp/karinako/default_run_long/0/ocn_ts.nc"

# Fixed CLIMBER-X parameters
const CLIMBER_FIXED_PARAMS = Dict(
    "ctl.nyears" => 7000,
    "ctl.co2_const" => 190,
    "ctl.fake_geo_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.fake_ice_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.restart_in_dir" => "/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000",
    "ocn.l_noise_fw" => "T",
    "ocn.noise_amp_fw" => 0.4
)

# Calibration parameters (to be varied)
const PARAM_NAMES = [
    "diff_dia_min",
    "drag_topo_fac", 
    "slope_max",
    "diff_iso",
    "diff_gm",
    "diff_dia_max"
]

# Prior bounds (uniform distributions)
const PRIOR_BOUNDS = Dict(
    "diff_dia_min" => (6e-6, 1.4e-5),
    "drag_topo_fac" => (2.6, 3.4),
    "slope_max" => (6e-4, 1.4e-3),
    "diff_iso" => (1100.0, 1900.0),
    "diff_gm" => (1100.0, 1900.0),
    "diff_dia_max" => (1.1e-4, 1.9e-4)
)

# PDF calibration settings
const PDF_GRID_POINTS = 100
const PDF_TOLERANCE = 0.03  # L2 distance tolerance for PDF (equivalent to ~0.01 supremum)

# Dynamical statistics uncertainties (from your original setup)
const WAITING_TIME_UNCERTAINTY = 39.1  # years
const STADIAL_DURATION_UNCERTAINTY = 42.6  # years

# ============================================
# NORMALIZATION UTILITIES
# ============================================

"""
Normalize observations by their uncertainties
"""
function normalize_observations(y, uncertainties)
    return y ./ uncertainties
end

"""
Denormalize observations back to physical units
"""
function denormalize_observations(y_normalized, uncertainties)
    return y_normalized .* uncertainties
end

# ============================================
# JOB SUBMISSION USING RUNME
# ============================================

"""
Submit a CLIMBER-X job using runme -s (submit mode)
Returns the job ID and expected output file path
"""
function submit_climber_job_with_runme(iteration, member_id, params_dict, output_dir, work_dir; 
                                       walltime="20:00:00", qos="standby", omp=32)
    # Output directory for this member
    member_output_dir = joinpath(output_dir, "iter_$(iteration)", "member_$(member_id)")
    
    # Expected output file
    output_file = joinpath(member_output_dir, "ocn_ts.nc")
    
    # Change to CLIMBER-X directory to run runme
    original_dir = pwd()
    cd(CLIMBER_X_DIR)
    
    try
        # Build parameter string exactly like the bash script
        param_str = ""
        for (key, val) in params_dict
            param_str *= " $(key)=$(val)"
        end
        
        # Construct the full command as a shell string
        cmd_str = """./runme -rs -q $(qos) -w $(walltime) --omp $(omp) -o "$(member_output_dir)" -p$(param_str)"""
        
        println("    Submitting member $member_id with command:")
        println("      $cmd_str")
        
        # Execute via shell
        output = read(`bash -c $cmd_str`, String)
        
        # Extract job ID from output
        job_id_match = match(r"Submitted batch job (\d+)", output)
        if job_id_match !== nothing
            job_id = job_id_match.captures[1]
            cd(original_dir)
            return job_id, output_file
        else
            @warn "Could not extract job ID from runme output for member $member_id"
            @warn "Output was: $output"
            cd(original_dir)
            error("Failed to extract job ID")
        end
        
    catch e
        cd(original_dir)
        @error "Failed to submit job for member $member_id" exception=e
        rethrow(e)
    end
end

"""
Submit CLIMBER-X jobs for one iteration using runme
"""
function submit_iteration_jobs_climber(params_i, iteration, work_dir, output_dir)
    N_ensemble = size(params_i, 2)
    job_trackers = JobTracker[]
    
    println("\n  Submitting $N_ensemble CLIMBER-X jobs for iteration $iteration...")
    println("  Using runme -rs to submit jobs")
    println("  Expected runtime: ~20 hours per job (10000 years)")
    
    # Check disk space
    has_space, available_gb = check_disk_space(output_dir, min_gb_required=100, warn_gb=500)
    if !has_space
        error("Insufficient disk space")
    end
    
    # Submit jobs
    for j in 1:N_ensemble
        # Build parameter dictionary for this member
        params_dict = Dict{String, Any}()
        
        # Add calibration parameters (with ocn. prefix)
        for (idx, name) in enumerate(PARAM_NAMES)
            params_dict["ocn.$(name)"] = params_i[idx, j]
        end
        
        # Add fixed parameters
        for (key, val) in CLIMBER_FIXED_PARAMS
            params_dict[key] = val
        end
        
        # Submit job
        try
            job_id, output_file = submit_climber_job_with_runme(
                iteration, j, params_dict, output_dir, work_dir;
                qos="standby",
                walltime="20:00:00"
            )
            
            tracker = JobTracker(
                job_id,
                j,
                iteration,
                :submitted,
                now(),
                nothing,
                "",
                output_file
            )
            push!(job_trackers, tracker)
            
            if j % 10 == 0 || j == N_ensemble
                println("    Submitted $j/$N_ensemble jobs")
            end
            
            sleep(2)  # Rate limiting
            
        catch e
            @error "Failed to submit member $j" exception=e
        end
    end
    
    if length(job_trackers) < N_ensemble
        @warn "Only submitted $(length(job_trackers))/$N_ensemble jobs successfully"
    else
        println("  ✓ All $N_ensemble jobs submitted!")
    end
    
    return job_trackers
end

# ============================================
# OUTPUT VALIDATION
# ============================================

"""
Validate CLIMBER-X output file
"""
function validate_climber_output_file(output_file; min_size_bytes=100000)
    if !isfile(output_file)
        return false, "File does not exist"
    end
    
    file_size = filesize(output_file)
    if file_size < min_size_bytes
        return false, "File too small: $(file_size) bytes"
    end
    
    try
        ds = NCDataset(output_file)
        has_amoc = haskey(ds, "amoc26N")
        has_time = haskey(ds, "time")
        close(ds)
        
        if !has_amoc
            return false, "Missing amoc26N variable"
        end
        if !has_time
            return false, "Missing time variable"
        end
        
        return true, "Valid"
    catch e
        return false, "Cannot read NetCDF: $e"
    end
end

# ============================================
# PDF AND STATISTICS COMPUTATION
# ============================================

"""
Compute PDF on a common grid for consistent comparison
"""
function compute_pdf_on_grid(amoc_data, x_grid; remove_spinup=true, spinup_fraction=0.02)
    amoc_data = vec(amoc_data)
    
    # Remove spinup
    if remove_spinup
        start_idx = Int(floor(length(amoc_data) * spinup_fraction)) + 1
        amoc_data = amoc_data[start_idx:end]
    end
    
    # Compute KDE
    kde_obj = kde(amoc_data)
    
    # Evaluate on grid
    pdf_vals = pdf(kde_obj, x_grid)
    
    # Normalize
    integral = sum((pdf_vals[1:end-1] .+ pdf_vals[2:end]) .* diff(x_grid)) / 2
    pdf_vals = pdf_vals ./ integral
    
    return pdf_vals
end

"""
Compute L2 distance between two PDFs
"""
function l2_distance(pdf1, pdf2, dx)
    # L2 norm: sqrt(∫(f1 - f2)² dx)
    # Discrete approximation: sqrt(Σ(f1 - f2)² * dx)
    diff = pdf1 .- pdf2
    return sqrt(sum(diff.^2) * dx)
end

"""
Read AMOC from CLIMBER-X output file
"""
function read_climber_amoc(output_file::String)
    if !isfile(output_file)
        error("Output file does not exist: $output_file")
    end
    
    try
        ds = NCDataset(output_file)
        amoc = ds["amoc26N"][:]
        time = ds["time"][:]
        close(ds)
        
        return amoc, time
    catch e
        @error "Failed to read CLIMBER-X output: $output_file" exception=e
        rethrow(e)
    end
end

"""
Process CLIMBER-X output and extract PDF + dynamical statistics
Returns: [pdf_values..., avg_waiting_time, avg_stadial_duration]
"""
function process_climber_output_with_stats(output_file::String, pdf_grid; 
                                          remove_spinup=true, spinup_fraction=0.02,
                                          do_min_spacing=500, do_crossing_value=5.0)
    # Read AMOC
    amoc, time = read_climber_amoc(output_file)
    
    # Compute PDF on common grid
    pdf_vals = compute_pdf_on_grid(amoc, pdf_grid, 
                                   remove_spinup=remove_spinup, 
                                   spinup_fraction=spinup_fraction)
    
    # Compute summary statistics (from your climber_summary_stats.jl)
    stats = compute_summary_stats(amoc; 
                                  time_data=time,
                                  remove_spinup=remove_spinup,
                                  spinup_fraction=spinup_fraction,
                                  adaptive_threshold=true,
                                  threshold_method="clustering",
                                  grid_points=length(pdf_grid),
                                  ignore_first_stadial=true,
                                  loess_span=0.02,
                                  do_min_spacing=do_min_spacing,
                                  do_crossing_value=do_crossing_value)
    
    # Combine PDF + dynamical statistics
    calibration_vector = vcat(
        pdf_vals,                          # 100 values
        stats["avg_waiting_time"],         # 1 value
        stats["avg_stadial_duration"]      # 1 value
    )
    
    return calibration_vector, stats
end

# ============================================
# RESULT COLLECTION
# ============================================

"""
Collect results from CLIMBER-X iteration using PDF + dynamical statistics
"""
function collect_climber_iteration_results(job_trackers, pdf_grid, y_obs, uncertainties; max_failures_allowed=5)
    N_ensemble = length(job_trackers)
    n_outputs = length(y_obs)  # PDF grid points + 2 dynamical stats
    G_ensemble = zeros(n_outputs, N_ensemble)
    
    n_failures = 0
    
    println("\n  Collecting CLIMBER-X results from $N_ensemble jobs...")
    
    for (j, tracker) in enumerate(job_trackers)
        if tracker.status == :completed
            is_valid, msg = validate_climber_output_file(tracker.output_file)
            
            if is_valid
                try
                    # Process output: get PDF + stats
                    calibration_vector, stats = process_climber_output_with_stats(
                        tracker.output_file, pdf_grid,
                        remove_spinup=true, 
                        spinup_fraction=0.02,
                        do_min_spacing=500,
                        do_crossing_value=5.0
                    )
                    
                    # Normalize by uncertainties
                    G_ensemble[:, j] = normalize_observations(calibration_vector, uncertainties)
                    
                    if j % 10 == 0
                        println("    Processed $j/$N_ensemble outputs")
                    end
                    
                catch e
                    @warn "Failed to process member $(tracker.member_id): $e"
                    G_ensemble[:, j] .= NaN
                    n_failures += 1
                end
            else
                @warn "Member $(tracker.member_id) output invalid: $msg"
                G_ensemble[:, j] .= NaN
                n_failures += 1
            end
        else
            @warn "Member $(tracker.member_id) did not complete (status: $(tracker.status))"
            G_ensemble[:, j] .= NaN
            n_failures += 1
        end
    end
    
    if n_failures > max_failures_allowed
        @error "Too many failures: $n_failures/$N_ensemble (max allowed: $max_failures_allowed)"
        error("Iteration failed due to excessive failures")
    elseif n_failures > 0
        @warn "$n_failures/$N_ensemble members failed"
    end
    
    println("  ✓ Results collected: $(N_ensemble - n_failures) successful")
    
    # Compute and report statistics (denormalize for reporting)
    valid_members = [j for j in 1:N_ensemble if !any(isnan.(G_ensemble[:, j]))]
    if !isempty(valid_members)
        n_pdf = length(pdf_grid)
        dx = step(pdf_grid)
        
        # Denormalize for physical interpretation
        G_denorm = zeros(n_outputs, length(valid_members))
        for (idx, j) in enumerate(valid_members)
            G_denorm[:, idx] = denormalize_observations(G_ensemble[:, j], uncertainties)
        end
        
        y_obs_denorm = denormalize_observations(y_obs, uncertainties)
        
        # PDF L2 distances (in physical units)
        pdf_obs = y_obs_denorm[1:n_pdf]
        l2_distances = [l2_distance(G_denorm[1:n_pdf, idx], pdf_obs, dx) for idx in 1:length(valid_members)]
        
        # Waiting times (in physical units)
        waiting_times = G_denorm[n_pdf+1, :]
        waiting_time_obs = y_obs_denorm[n_pdf+1]
        
        # Stadial durations (in physical units)
        stadial_durations = G_denorm[n_pdf+2, :]
        stadial_duration_obs = y_obs_denorm[n_pdf+2]
        
        println("\n  PDF L2 distance statistics (physical units):")
        println("    Mean: $(round(mean(l2_distances), digits=6))")
        println("    Min:  $(round(minimum(l2_distances), digits=6))")
        println("    Max:  $(round(maximum(l2_distances), digits=6))")
        println("    Members within tolerance (< $PDF_TOLERANCE): $(sum(l2_distances .< PDF_TOLERANCE))/$(length(l2_distances))")
        
        println("\n  Waiting time statistics (years):")
        println("    Target: $(round(waiting_time_obs, digits=1)) years")
        println("    Mean:   $(round(mean(waiting_times), digits=1)) years")
        println("    Std:    $(round(std(waiting_times), digits=1)) years")
        println("    Range:  [$(round(minimum(waiting_times), digits=1)), $(round(maximum(waiting_times), digits=1))] years")
        
        println("\n  Stadial duration statistics (years):")
        println("    Target: $(round(stadial_duration_obs, digits=1)) years")
        println("    Mean:   $(round(mean(stadial_durations), digits=1)) years")
        println("    Std:    $(round(std(stadial_durations), digits=1)) years")
        println("    Range:  [$(round(minimum(stadial_durations), digits=1)), $(round(maximum(stadial_durations), digits=1))] years")
        
        println("\n  Normalized observation statistics (for EKI):")
        println("    PDF components: $(round(mean(G_ensemble[1:n_pdf, valid_members]), digits=3)) ± $(round(std(G_ensemble[1:n_pdf, valid_members]), digits=3))")
        println("    Waiting time: $(round(mean(G_ensemble[n_pdf+1, valid_members]), digits=3)) ± $(round(std(G_ensemble[n_pdf+1, valid_members]), digits=3))")
        println("    Stadial duration: $(round(mean(G_ensemble[n_pdf+2, valid_members]), digits=3)) ± $(round(std(G_ensemble[n_pdf+2, valid_members]), digits=3))")
    end
    
    return G_ensemble
end

"""
Collect results from existing output files (for resuming)
"""
function collect_results_from_files(output_dir, iteration, N_ensemble, pdf_grid, y_obs, uncertainties; max_failures_allowed=5)
    n_outputs = length(y_obs)
    G_ensemble = zeros(n_outputs, N_ensemble)
    n_failures = 0
    
    println("\n  Collecting results from existing files for iteration $iteration...")
    
    for j in 1:N_ensemble
        output_file = joinpath(output_dir, "iter_$(iteration)", "member_$(j)", "ocn_ts.nc")
        is_valid, msg = validate_climber_output_file(output_file)
        
        if is_valid
            try
                calibration_vector, _ = process_climber_output_with_stats(
                    output_file, pdf_grid,
                    remove_spinup=true, 
                    spinup_fraction=0.02,
                    do_min_spacing=500,
                    do_crossing_value=5.0
                )
                G_ensemble[:, j] = normalize_observations(calibration_vector, uncertainties)
                
                if j % 10 == 0
                    println("    Processed $j/$N_ensemble outputs")
                end
            catch e
                @warn "Failed to process member $j: $e"
                G_ensemble[:, j] .= NaN
                n_failures += 1
            end
        else
            @warn "Member $j output invalid: $msg"
            G_ensemble[:, j] .= NaN
            n_failures += 1
        end
    end
    
    if n_failures > max_failures_allowed
        error("Too many failures: $n_failures/$N_ensemble")
    end
    
    println("  ✓ Results collected: $(N_ensemble - n_failures) successful")
    return G_ensemble
end

# ============================================
# CHECKPOINT MANAGEMENT
# ============================================

"""
Save checkpoint
"""
function save_checkpoint(iteration, eksobj, prior, param_history, 
                        y_obs, obs_noise_cov, pdf_grid, uncertainties, metadata, checkpoint_dir)
    checkpoint_file = joinpath(checkpoint_dir, "checkpoint_iter_$(iteration).jld2")
    
    checkpoint_data = Dict(
        "iteration" => iteration,
        "eksobj" => eksobj,
        "prior" => prior,
        "param_history" => param_history,
        "y_obs" => y_obs,
        "obs_noise_cov" => obs_noise_cov,
        "pdf_grid" => pdf_grid,
        "uncertainties" => uncertainties,
        "metadata" => metadata
    )
    
    @save checkpoint_file checkpoint_data
    println("  ✓ Checkpoint saved: $checkpoint_file")
end

"""
Save iteration results with L2 distances and statistics
"""
function save_iteration_results(iteration, params_i, G_ensemble, 
                               current_mean, current_std, y_obs, pdf_grid, uncertainties, output_dir)
    results_file = joinpath(output_dir, "iteration_$(iteration)_results.jld2")
    
    # Compute diagnostics for all valid members (in physical units)
    dx = step(pdf_grid)
    n_pdf = length(pdf_grid)
    
    l2_distances = Float64[]
    waiting_times = Float64[]
    stadial_durations = Float64[]
    
    y_obs_denorm = denormalize_observations(y_obs, uncertainties)
    
    for j in 1:size(G_ensemble, 2)
        if !any(isnan.(G_ensemble[:, j]))
            G_denorm = denormalize_observations(G_ensemble[:, j], uncertainties)
            push!(l2_distances, l2_distance(G_denorm[1:n_pdf], y_obs_denorm[1:n_pdf], dx))
            push!(waiting_times, G_denorm[n_pdf+1])
            push!(stadial_durations, G_denorm[n_pdf+2])
        else
            push!(l2_distances, NaN)
            push!(waiting_times, NaN)
            push!(stadial_durations, NaN)
        end
    end
    
    @save results_file params_i G_ensemble current_mean current_std l2_distances waiting_times stadial_durations uncertainties
    
    println("  ✓ Iteration results saved: $results_file")
end

"""
Save final results
"""
function save_final_results(θ_optimal, θ_std, final_ensemble, 
                          y_obs, pdf_grid, uncertainties, metadata, output_dir)
    final_file = joinpath(output_dir, "final_results.jld2")
    
    @save final_file θ_optimal θ_std final_ensemble y_obs pdf_grid uncertainties metadata
    
    println("  ✓ Final results saved: $final_file")
end

# ============================================
# MAIN CALIBRATION FUNCTION
# ============================================

function run_climber_x_calibration(;
    N_iterations=10,
    N_ensemble=50,
    output_dir="/p/tmp/karinako/eki_calibration/output",
    work_dir="/p/tmp/karinako/eki_calibration/working",
    check_interval_minutes=30,
    max_wait_days=10,
    pdf_grid_points=100)
    
    println("="^80)
    println("CLIMBER-X EKI CALIBRATION - PDF + DYNAMICAL STATISTICS (NORMALIZED)")
    println("="^80)
    println("Parameters: $(length(PARAM_NAMES)) ocean parameters")
    println("Observations:")
    println("  - PDF with $pdf_grid_points grid points (L2 tolerance: $PDF_TOLERANCE)")
    println("  - Average waiting time (σ = $WAITING_TIME_UNCERTAINTY years)")
    println("  - Average stadial duration (σ = $STADIAL_DURATION_UNCERTAINTY years)")
    println("  - All observations normalized by uncertainties for balanced fitting")
    println("Ensemble size: $N_ensemble")
    println("Iterations: $N_iterations")
    println("Output directory: $output_dir")
    println("="^80)
    
    # Check that runme script exists
    if !isfile(RUNME_SCRIPT)
        error("CLIMBER-X runme script not found: $RUNME_SCRIPT")
    end
    println("  ✓ Found runme script: $RUNME_SCRIPT")
    
    # Check Python and runner module availability
    println("\nChecking Python environment...")
    try
        run(`python3 -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'; import runner"`)
        println("  ✓ Python 3.8+ and runner module available")
    catch e
        @error "Python 3.8+ or runner module not found"
        rethrow(e)
    end
    
    # Create directories
    mkpath(output_dir)
    mkpath(work_dir)
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    mkpath(checkpoint_dir)
    
    # Check disk space
    has_space, _ = check_disk_space(output_dir, min_gb_required=500, warn_gb=1000)
    if !has_space
        error("Insufficient disk space")
    end
    
    println("\nSetting up prior distributions...")

    # Create parameter distributions - CORRECTED VERSION
    prior_dists = ParameterDistribution[]

    for name in PARAM_NAMES
        bounds = PRIOR_BOUNDS[name]
        lower = bounds[1]
        upper = bounds[2]
        
        # Use Parameterized with Uniform (NO additional constraint needed)
        dist = Parameterized(Uniform(lower, upper))
        push!(prior_dists, ParameterDistribution(dist, no_constraint(), name))
    end

    prior = combine_distributions(prior_dists)

    # Add diagnostic
    println("\n  Verifying prior samples:")
    test_ensemble = construct_initial_ensemble(prior, 5)
    for (idx, name) in enumerate(PARAM_NAMES)
        println("    $(name): $(test_ensemble[idx, :])")
    end
    
    # Check for existing checkpoint to resume from
    start_iteration = 1
    eksobj = nothing
    y_obs = nothing
    pdf_grid = nothing
    param_history = nothing
    uncertainties = nothing
    
    # Find latest checkpoint
    existing_checkpoints = filter(f -> startswith(f, "checkpoint_iter_") && endswith(f, ".jld2"), 
                                  readdir(checkpoint_dir))
    
    if !isempty(existing_checkpoints)
        # Extract iteration numbers and find max
        iter_nums = [parse(Int, match(r"checkpoint_iter_(\d+)\.jld2", f).captures[1]) 
                     for f in existing_checkpoints]
        latest_iter = maximum(iter_nums)
        
        if latest_iter > 0
            println("\n  Found checkpoint at iteration $latest_iter")
            print("  Resume from checkpoint? (y/n): ")
            response = readline()
            
            if lowercase(strip(response)) == "y"
                checkpoint_file = joinpath(checkpoint_dir, "checkpoint_iter_$(latest_iter).jld2")
                @load checkpoint_file checkpoint_data
                
                eksobj = checkpoint_data["eksobj"]
                prior = checkpoint_data["prior"]
                param_history = checkpoint_data["param_history"]
                y_obs = checkpoint_data["y_obs"]
                pdf_grid = checkpoint_data["pdf_grid"]
                uncertainties = checkpoint_data["uncertainties"]
                
                start_iteration = latest_iter + 1
                println("  ✓ Resuming from iteration $start_iteration")
            end
        end
    end
    
    # If not resuming, initialize fresh
    if isnothing(eksobj)
        # Process default run to get observations
        println("\nProcessing default run for target observations...")
        println("  Default run: $DEFAULT_RUN_OUTPUT")
        
        if !isfile(DEFAULT_RUN_OUTPUT)
            error("Default run output not found: $DEFAULT_RUN_OUTPUT")
        end
        
        # Read default run
        amoc_default, time_default = read_climber_amoc(DEFAULT_RUN_OUTPUT)
        
        # Remove spinup
        start_idx = Int(floor(length(amoc_default) * 0.02)) + 1
        amoc_default = amoc_default[start_idx:end]
        time_default = time_default[start_idx:end]
        
        # Create common grid spanning reasonable AMOC range
        x_min = minimum(amoc_default) - 2.0
        x_max = maximum(amoc_default) + 2.0
        pdf_grid = range(x_min, x_max, length=pdf_grid_points)
        
        # Compute default PDF on this grid
        pdf_obs = compute_pdf_on_grid(amoc_default, pdf_grid, remove_spinup=false)
        
        # Compute dynamical statistics from default run
        stats_default = compute_summary_stats(amoc_default; 
                                             time_data=time_default,
                                             remove_spinup=false,
                                             spinup_fraction=0.0,
                                             adaptive_threshold=true,
                                             threshold_method="clustering",
                                             loess_span=0.02,
                                             do_min_spacing=500,
                                             do_crossing_value=5.0)
        
        # Create raw observation vector
        y_obs_raw = vcat(
            pdf_obs,                                    # 100 values
            stats_default["avg_waiting_time"],          # 1 value
            stats_default["avg_stadial_duration"]       # 1 value
        )
        
        dx = step(pdf_grid)
        
        # Create uncertainty vector
        pdf_obs_uncertainty = PDF_TOLERANCE / sqrt(pdf_grid_points * dx)
        uncertainties = vcat(
            fill(pdf_obs_uncertainty, pdf_grid_points),
            WAITING_TIME_UNCERTAINTY,
            STADIAL_DURATION_UNCERTAINTY
        )
        
        # Normalize observations by uncertainties
        y_obs = normalize_observations(y_obs_raw, uncertainties)
        
        println("  PDF grid: $(length(pdf_grid)) points from $(round(x_min, digits=2)) to $(round(x_max, digits=2))")
        println("  Grid spacing (dx): $(round(dx, digits=4))")
        println("\n  Target observations (physical units):")
        println("    PDF max: $(round(maximum(pdf_obs), digits=4))")
        println("    PDF integral: $(round(sum((pdf_obs[1:end-1] .+ pdf_obs[2:end]) .* diff(pdf_grid))/2, digits=4))")
        println("    Avg waiting time: $(round(stats_default["avg_waiting_time"], digits=1)) years")
        println("    Avg stadial duration: $(round(stats_default["avg_stadial_duration"], digits=1)) years")
        println("    N DO events: $(stats_default["n_do_events"])")
        println("    N stadials: $(stats_default["n_stadials"])")
        
        println("\n  Target observations (normalized):")
        println("    PDF range: [$(round(minimum(y_obs[1:pdf_grid_points]), digits=3)), $(round(maximum(y_obs[1:pdf_grid_points]), digits=3))]")
        println("    Waiting time: $(round(y_obs[pdf_grid_points+1], digits=3))")
        println("    Stadial duration: $(round(y_obs[pdf_grid_points+2], digits=3))")
        
        println("\n  Observation uncertainties:")
        println("    PDF (per grid point): $(round(pdf_obs_uncertainty, digits=6))")
        println("    Waiting time: $(WAITING_TIME_UNCERTAINTY) years")
        println("    Stadial duration: $(STADIAL_DURATION_UNCERTAINTY) years")
        
        # Use unit observation covariance (since observations are normalized)
        obs_noise_cov = Diagonal(ones(length(y_obs)))
        
        println("\n  Using unit observation covariance (normalized observations)")
        
        println("\nInitializing EKI process...")
        initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
        eks_process = Sampler(prior)

        println("\n  DEBUG: Initial ensemble after construction:")
        for j in 1:min(3, N_ensemble)
            println("  Member $j:")
            for (idx, name) in enumerate(PARAM_NAMES)
                println("    $(name): $(initial_ensemble[idx, j])")
            end
        end
        
        eksobj = EnsembleKalmanProcess(
            initial_ensemble,
            y_obs,
            obs_noise_cov,
            eks_process,
            verbose=true
        )

        # DEBUG: Verify the ensemble after EKI initialization
        params_after_init = get_ϕ_final(prior, eksobj)
        println("\n  DEBUG: Ensemble after EKI initialization:")
        for j in 1:min(3, N_ensemble)
            println("  Member $j:")
            for (idx, name) in enumerate(PARAM_NAMES)
                println("    $(name): $(params_after_init[idx, j])")
            end
        end
        # Check if they match
        println("\n  DEBUG: Comparing initial_ensemble vs get_ϕ_final:")
        for j in 1:min(3, N_ensemble)
            println("  Member $j max difference: $(maximum(abs.(initial_ensemble[:, j] .- params_after_init[:, j])))")
        end
        
        param_history = zeros(length(PARAM_NAMES), N_iterations + 1, N_ensemble)
        param_history[:, 1, :] = get_ϕ_final(prior, eksobj)
        
        metadata = Dict(
            "start_time" => now(),
            "N_iterations" => N_iterations,
            "N_ensemble" => N_ensemble,
            "param_names" => PARAM_NAMES,
            "pdf_grid_points" => pdf_grid_points,
            "pdf_tolerance" => PDF_TOLERANCE,
            "waiting_time_uncertainty" => WAITING_TIME_UNCERTAINTY,
            "stadial_duration_uncertainty" => STADIAL_DURATION_UNCERTAINTY,
            "default_run" => DEFAULT_RUN_OUTPUT,
            "distance_metric" => "L2",
            "observations" => "PDF + waiting_time + stadial_duration (normalized)",
            "normalization" => "by_uncertainty"
        )
        
        save_checkpoint(0, eksobj, prior, param_history, 
                       y_obs, obs_noise_cov, pdf_grid, uncertainties, metadata, checkpoint_dir)
    end
    
    obs_noise_cov = Diagonal(ones(length(y_obs)))
    
    metadata = Dict(
        "start_time" => now(),
        "N_iterations" => N_iterations,
        "N_ensemble" => N_ensemble,
        "param_names" => PARAM_NAMES,
        "pdf_grid_points" => pdf_grid_points,
        "pdf_tolerance" => PDF_TOLERANCE,
        "waiting_time_uncertainty" => WAITING_TIME_UNCERTAINTY,
        "stadial_duration_uncertainty" => STADIAL_DURATION_UNCERTAINTY,
        "default_run" => DEFAULT_RUN_OUTPUT,
        "distance_metric" => "L2",
        "observations" => "PDF + waiting_time + stadial_duration (normalized)",
        "normalization" => "by_uncertainty"
    )
    
    # Main iteration loop
    for i in start_iteration:N_iterations
        iter_start_time = now()
        
        println("\n" * "="^80)
        println("ITERATION $i/$N_iterations")
        println("="^80)
        
        params_i = get_ϕ_final(prior, eksobj)

        # Right after: params_i = get_ϕ_final(prior, eksobj)

        println("\n  DEBUG: First 3 ensemble members:")
        for j in 1:min(3, size(params_i, 2))
            println("  Member $j:")
            for (idx, name) in enumerate(PARAM_NAMES)
                println("    $(name): $(params_i[idx, j])")
            end
        end
        
        # Check if iteration already has completed outputs
        iter_dir = joinpath(output_dir, "iter_$(i)")
        all_outputs_exist = true
        if isdir(iter_dir)
            for j in 1:N_ensemble
                output_file = joinpath(iter_dir, "member_$(j)", "ocn_ts.nc")
                if !validate_climber_output_file(output_file)[1]
                    all_outputs_exist = false
                    break
                end
            end
        else
            all_outputs_exist = false
        end
        
        if all_outputs_exist
            println("  Found existing outputs for iteration $i, skipping job submission...")
            
            # Create dummy job trackers with completed status
            job_trackers = JobTracker[]
            for j in 1:N_ensemble
                output_file = joinpath(iter_dir, "member_$(j)", "ocn_ts.nc")
                tracker = JobTracker("existing", j, i, :completed, now(), now(), "", output_file)
                push!(job_trackers, tracker)
            end
        else
            # Submit jobs
            job_trackers = submit_iteration_jobs_climber(
                params_i, i, work_dir, output_dir
            )
            
            save_job_trackers(job_trackers, i, output_dir)
            
            # Wait for completion
            result = wait_for_iteration_completion(
                job_trackers;
                check_interval_minutes=check_interval_minutes,
                max_wait_days=max_wait_days,
                output_dir=output_dir
            )
            
            if result == :timeout
                error("Iteration $i timed out")
            end
        end
        
        # Collect results (returns normalized G_ensemble)
        G_ensemble = collect_climber_iteration_results(job_trackers, pdf_grid, y_obs, uncertainties,
                                                       max_failures_allowed=5)
        
        # Update ensemble
        println("\n  Updating ensemble with EKI...")
        update_ensemble!(eksobj, G_ensemble)
        param_history[:, i+1, :] = get_ϕ_final(prior, eksobj)
        
        # Current parameter estimates
        current_mean = get_ϕ_mean_final(prior, eksobj)
        current_std = std(get_ϕ_final(prior, eksobj), dims=2)
        
        println("\n  Current parameter estimates:")
        for (idx, name) in enumerate(PARAM_NAMES)
            bounds = PRIOR_BOUNDS[name]
            println("    $(rpad(name, 20)): $(round(current_mean[idx], sigdigits=4)) ± $(round(current_std[idx], sigdigits=3)) (bounds: $(bounds))")
        end
        
        # Iteration duration
        iter_duration = now() - iter_start_time
        println("\n  Iteration duration: $(iter_duration)")
        
        # Save results
        save_iteration_results(i, params_i, G_ensemble, 
                             current_mean, current_std, y_obs, pdf_grid, uncertainties, output_dir)
        
        save_checkpoint(i, eksobj, prior, param_history,
                       y_obs, obs_noise_cov, pdf_grid, uncertainties, metadata, checkpoint_dir)
    end
    
    # Final results
    θ_optimal = get_ϕ_mean_final(prior, eksobj)
    final_ensemble = get_ϕ_final(prior, eksobj)
    θ_std = std(final_ensemble, dims=2)
    
    save_final_results(θ_optimal, vec(θ_std), final_ensemble,
                      y_obs, pdf_grid, uncertainties, metadata, output_dir)
    
    println("\n" * "="^80)
    println("CLIMBER-X CALIBRATION COMPLETE!")
    println("="^80)
    
    println("\nFinal parameter estimates:")
    println("-"^80)
    @printf("%-20s | %15s | %12s | %15s\n", "Parameter", "Optimized", "Std Dev", "Prior Bounds")
    println("-"^80)
    for (idx, name) in enumerate(PARAM_NAMES)
        bounds = PRIOR_BOUNDS[name]
        @printf("%-20s | %15.6g | %12.6g | [%.6g, %.6g]\n", 
                name, θ_optimal[idx], θ_std[idx], bounds[1], bounds[2])
    end
    println("-"^80)
    
    # Compute final statistics
    valid_members = []
    final_pdfs = []
    final_waiting_times = []
    final_stadial_durations = []
    
    n_pdf = length(pdf_grid)
    
    for j in 1:N_ensemble
        output_file = joinpath(output_dir, "iter_$(N_iterations)", "member_$(j)", "ocn_ts.nc")
        if isfile(output_file) && validate_climber_output_file(output_file)[1]
            try
                calibration_vector, _ = process_climber_output_with_stats(
                    output_file, pdf_grid,
                    remove_spinup=true, 
                    spinup_fraction=0.02,
                    do_min_spacing=500,
                    do_crossing_value=5.0
                )
                push!(valid_members, j)
                push!(final_pdfs, calibration_vector[1:n_pdf])
                push!(final_waiting_times, calibration_vector[n_pdf+1])
                push!(final_stadial_durations, calibration_vector[n_pdf+2])
            catch e
                @warn "Could not process final member $j"
            end
        end
    end
    
    if !isempty(final_pdfs)
        dx = step(pdf_grid)
        y_obs_denorm = denormalize_observations(y_obs, uncertainties)
        pdf_obs = y_obs_denorm[1:n_pdf]
        
        l2_distances = [l2_distance(pdf, pdf_obs, dx) for pdf in final_pdfs]
        
        println("\nFinal PDF matching performance:")
        println("  L2 distance - Mean: $(round(mean(l2_distances), digits=6))")
        println("  L2 distance - Min:  $(round(minimum(l2_distances), digits=6))")
        println("  L2 distance - Max:  $(round(maximum(l2_distances), digits=6))")
        println("  Members within tolerance (< $PDF_TOLERANCE): $(sum(l2_distances .< PDF_TOLERANCE))/$(length(l2_distances))")
        
        println("\nFinal waiting time performance:")
        println("  Target: $(round(y_obs_denorm[n_pdf+1], digits=1)) years")
        println("  Mean:   $(round(mean(final_waiting_times), digits=1)) years")
        println("  Std:    $(round(std(final_waiting_times), digits=1)) years")
        println("  Range:  [$(round(minimum(final_waiting_times), digits=1)), $(round(maximum(final_waiting_times), digits=1))] years")
        
        println("\nFinal stadial duration performance:")
        println("  Target: $(round(y_obs_denorm[n_pdf+2], digits=1)) years")
        println("  Mean:   $(round(mean(final_stadial_durations), digits=1)) years")
        println("  Std:    $(round(std(final_stadial_durations), digits=1)) years")
        println("  Range:  [$(round(minimum(final_stadial_durations), digits=1)), $(round(maximum(final_stadial_durations), digits=1))] years")
    end
    
    return eksobj, param_history, metadata, pdf_grid, uncertainties
end

# ============================================
# RUN THE CALIBRATION
# ============================================

eksobj, param_history, metadata, pdf_grid, uncertainties = run_climber_x_calibration(
    N_iterations=6,
    N_ensemble=60,
    output_dir="/p/tmp/karinako/eki_calibration_7000/output",
    work_dir="/p/tmp/karinako/eki_calibration_7000/working",
    check_interval_minutes=30,
    max_wait_days=10,
    pdf_grid_points=100
)
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
# Observation uncertainties (matching MCMC/ABC setup)
const OBS_UNCERTAINTIES = [
    0.18,   # PCA component 1 (same σ for all 5 components)
    0.18,   # PCA component 2
    0.18,   # PCA component 3
    0.18,   # PCA component 4
    0.18,   # PCA component 5
    39.1,   # avg_waiting_time (years)
    42.6    # avg_stadial_duration (years)
]

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
# RESULT COLLECTION
# ============================================

"""
Collect results from CLIMBER-X iteration using PCA model
"""
function collect_climber_iteration_results(job_trackers, pca_model, y_obs; max_failures_allowed=5)
    N_ensemble = length(job_trackers)
    n_outputs = length(y_obs)
    G_ensemble = zeros(n_outputs, N_ensemble)
    
    n_failures = 0
    
    println("\n  Collecting CLIMBER-X results from $N_ensemble jobs...")
    
    for (j, tracker) in enumerate(job_trackers)
        if tracker.status == :completed
            is_valid, msg = validate_climber_output_file(tracker.output_file)
            
            if is_valid
                try
                    calibration_stats, full_stats = process_climber_output(
                        tracker.output_file, pca_model,
                        remove_spinup=true, spinup_fraction=0.02, do_min_spacing=500,
                        do_crossing_value=2.0
                    )
                    
                    G_ensemble[:, j] = calibration_stats
                    
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
    
    return G_ensemble
end

"""
Collect results from existing output files (for resuming)
"""
function collect_results_from_files(output_dir, iteration, N_ensemble, pca_model, y_obs; max_failures_allowed=5)
    n_outputs = length(y_obs)
    G_ensemble = zeros(n_outputs, N_ensemble)
    n_failures = 0
    
    println("\n  Collecting results from existing files for iteration $iteration...")
    
    for j in 1:N_ensemble
        output_file = joinpath(output_dir, "iter_$(iteration)", "member_$(j)", "ocn_ts.nc")
        is_valid, msg = validate_climber_output_file(output_file)
        
        if is_valid
            try
                calibration_stats, _ = process_climber_output(
                    output_file, pca_model,
                    remove_spinup=true, spinup_fraction=0.02, do_min_spacing=500, 
                    do_crossing_value = 2.0
                )
                G_ensemble[:, j] = calibration_stats
                
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
# PCA MODEL MANAGEMENT
# ============================================

"""
Load or fit PCA model
"""
function load_or_fit_pca(output_dir, ensemble_files, pca_components)
    pca_file = joinpath(output_dir, "pca_model.jld2")
    
    if isfile(pca_file)
        println("\n  Loading existing PCA model from $pca_file")
        @load pca_file pca_model y_obs
        println("  ✓ PCA model loaded")
        println("  Target observations:")
        println("    PCA components: $(round.(y_obs[1:5], digits=4))")
        println("    Avg waiting time: $(round(y_obs[6], digits=1)) years")
        println("    Avg stadial duration: $(round(y_obs[7], digits=1)) years")
        return pca_model, y_obs
    else
        println("\n  Fitting new PCA model...")
        
        if length(ensemble_files) < 5
            error("Cannot fit PCA with only $(length(ensemble_files)) successful runs. Need at least 5.")
        end
        
        pca_model, stats_default = initialize_pca_model(
            DEFAULT_RUN_OUTPUT,
            ensemble_files;
            n_components=pca_components,
            remove_spinup=true,
            spinup_fraction=0.02
        )
        
        # Extract target observations from default run
        default_calibration_stats, _ = process_climber_output(
            DEFAULT_RUN_OUTPUT, pca_model,
            remove_spinup=true, spinup_fraction=0.02,
        )
        y_obs = default_calibration_stats
        
        println("\n  Target observations (from default run):")
        println("    PCA components: $(round.(y_obs[1:5], digits=4))")
        println("    Avg waiting time: $(round(y_obs[6], digits=1)) years")
        println("    Avg stadial duration: $(round(y_obs[7], digits=1)) years")
        
        # Save PCA model for future use
        @save pca_file pca_model y_obs
        println("  ✓ PCA model saved to $pca_file")
        
        return pca_model, y_obs
    end
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
    pca_components=5,
    )
    
    println("="^80)
    println("CLIMBER-X EKI CALIBRATION")
    println("="^80)
    println("Parameters: $(length(PARAM_NAMES)) ocean parameters")
    println("Observations: $pca_components PCA components + 2 dynamical statistics")
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
    
    # Setup prior
    println("\nSetting up prior distributions...")

    prior_dists = ParameterDistribution[]
    for name in PARAM_NAMES
        bounds = PRIOR_BOUNDS[name]
        
        # Create uniform distribution using the built-in constructor
        # For uniform distribution in constrained space, use Parameterized with Uniform
        # and apply the bounded constraint
        dist = Parameterized(Uniform(bounds[1], bounds[2]))
        constraint = bounded(bounds[1], bounds[2])
        
        push!(prior_dists, 
            ParameterDistribution(dist, constraint, name))
    end
    prior = combine_distributions(prior_dists)

    println("Prior configured for $(length(PARAM_NAMES)) parameters with uniform distributions")
    
    # Check for existing checkpoint to resume from
    start_iteration = 1
    eksobj = nothing
    pca_model = nothing
    y_obs = nothing
    param_history = nothing
    
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
                
                # Load PCA model
                pca_file = joinpath(output_dir, "pca_model.jld2")
                if isfile(pca_file)
                    @load pca_file pca_model
                end
                
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
        
        amoc_default, time_default = read_climber_amoc(DEFAULT_RUN_OUTPUT)
        default_stats = compute_summary_stats(amoc_default; 
                                             time_data=time_default,
                                             remove_spinup=true,
                                             spinup_fraction=0.02)
        
        println("  Default run statistics:")
        println("    N stadials: $(default_stats["n_stadials"])")
        println("    Avg stadial duration: $(round(default_stats["avg_stadial_duration"], digits=1)) years")
        println("    N DO events: $(default_stats["n_do_events"])")
        println("    Avg waiting time: $(round(default_stats["avg_waiting_time"], digits=1)) years")
        
        # Initialize EKI with placeholder observations (will update after PCA)
        obs_noise_cov = Diagonal(OBS_UNCERTAINTIES.^2)
        
        println("\nInitializing EKI process...")
        initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
        eks_process = Sampler(prior)
        
        y_obs_placeholder = zeros(7)
        
        eksobj = EnsembleKalmanProcess(
            initial_ensemble,
            y_obs_placeholder,
            obs_noise_cov,
            eks_process,
            verbose=true
        )
        
        param_history = zeros(length(PARAM_NAMES), N_iterations + 1, N_ensemble)
        param_history[:, 1, :] = get_ϕ_final(prior, eksobj)
        
        metadata = Dict(
            "start_time" => now(),
            "N_iterations" => N_iterations,
            "N_ensemble" => N_ensemble,
            "param_names" => PARAM_NAMES,
            "obs_uncertainties" => OBS_UNCERTAINTIES,
            "default_run" => DEFAULT_RUN_OUTPUT,
        )
        
        save_checkpoint(0, eksobj, prior, param_history, 
                       y_obs_placeholder, obs_noise_cov, metadata, checkpoint_dir)
    end
    
    obs_noise_cov = Diagonal(OBS_UNCERTAINTIES.^2)
    
    metadata = Dict(
        "start_time" => now(),
        "N_iterations" => N_iterations,
        "N_ensemble" => N_ensemble,
        "param_names" => PARAM_NAMES,
        "obs_uncertainties" => OBS_UNCERTAINTIES,
        "default_run" => DEFAULT_RUN_OUTPUT,
    )
    
    # Main iteration loop
    for i in start_iteration:N_iterations
        iter_start_time = now()
        
        println("\n" * "="^80)
        println("ITERATION $i/$N_iterations")
        println("="^80)
        
        params_i = get_ϕ_final(prior, eksobj)
        
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
        
        # After first iteration: load or fit PCA
        if i == 1 && isnothing(pca_model)
            # Collect output files that completed successfully
            ensemble_files = [tracker.output_file for tracker in job_trackers 
                            if tracker.status == :completed && 
                               validate_climber_output_file(tracker.output_file)[1]]
            
            # Load or fit PCA model
            pca_model, y_obs = load_or_fit_pca(output_dir, ensemble_files, pca_components)
            
            # Recreate EKI with correct observations
            eks_process = Sampler(prior)
            eksobj = EnsembleKalmanProcess(
                get_ϕ(prior, eksobj, 1),  # FIXED: added prior argument
                y_obs,
                obs_noise_cov,
                eks_process,
                verbose=true
            )
        end
        
        # Collect results
        G_ensemble = collect_climber_iteration_results(job_trackers, pca_model, y_obs, 
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
                             current_mean, current_std, output_dir)
        
        save_checkpoint(i, eksobj, prior, param_history,
                       y_obs, obs_noise_cov, metadata, checkpoint_dir)
    end
    
    # Final results
    θ_optimal = get_ϕ_mean_final(prior, eksobj)
    final_ensemble = get_ϕ_final(prior, eksobj)
    θ_std = std(final_ensemble, dims=2)
    
    save_final_results(θ_optimal, vec(θ_std), final_ensemble,
                      y_obs, metadata, output_dir)
    
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
    
    return eksobj, param_history, metadata, pca_model
end

# ============================================
# RUN THE CALIBRATION
# ============================================

eksobj, param_history, metadata, pca_model = run_climber_x_calibration(
    N_iterations=6,
    N_ensemble=60,
    output_dir="/p/tmp/karinako/eki_calibration_7000/output",
    work_dir="/p/tmp/karinako/eki_calibration_7000/working",
    check_interval_minutes=30,
    max_wait_days=10,
    pca_components=5,
)
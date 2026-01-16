using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using JLD2
using Dates
using Printf
using Distributions  # for Uniform distribution in priors

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
    "ctl.nyears" => 75000,
    "ctl.co2_const" => 190,
    "ctl.fake_geo_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.fake_ice_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.restart_in_dir" => "/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000",
    "ocn.l_noise_fw" => "T"
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
    "diff_dia_min" => (7.5e-6, 1.25e-5),
    "drag_topo_fac" => (2.25, 3.75),
    "slope_max" => (7.5e-4, 1.25e-3),
    "diff_iso" => (1125.0, 1875.0),
    "diff_gm" => (1125.0, 1875.0),
    "diff_dia_max" => (1.125e-4, 1.875e-4)
)

# Observation uncertainties
const OBS_UNCERTAINTIES = [
    0.0189,  # PCA component 1
    0.0189,  # PCA component 2
    0.0189,  # PCA component 3
    0.0189,  # PCA component 4
    0.0189,  # PCA component 5
    39.1,    # avg_waiting_time (years)
    42.6     # avg_stadial_duration (years)
]

# ============================================
# PARAMETER FILE WRITING
# ============================================

"""
Write ensemble parameters to file in CLIMBER-X format
Format:
  Line 1: parameter names (with ocn. prefix)
  Line 2+: parameter values for each ensemble member
"""
function write_climber_params_for_iteration(params_i, iteration, work_dir)
    param_file = joinpath(work_dir, "iter_$(iteration)", "params_ensemble.txt")
    mkpath(dirname(param_file))
    
    # Add ocn. prefix to parameter names
    param_names_with_prefix = ["ocn." * name for name in PARAM_NAMES]
    
    N_ensemble = size(params_i, 2)
    
    open(param_file, "w") do f
        # Write header with parameter names
        println(f, join(param_names_with_prefix, " "))
        
        # Write each ensemble member as a row
        for j in 1:N_ensemble
            println(f, join([@sprintf("%.17g", val) for val in params_i[:, j]], " "))
        end
    end
    
    println("  ✓ Wrote parameter file: $param_file")
    return param_file
end

# ============================================
# CLIMBER-X JOB SCRIPT CREATION
# ============================================

"""
Create SLURM job script for one CLIMBER-X ensemble member
"""
function create_climber_job_script(iteration, member_id, param_file, output_dir, work_dir)
    # Output directory for this member
    member_output_dir = joinpath(output_dir, "iter_$(iteration)", "member_$(member_id)")
    mkpath(member_output_dir)
    
    # Expected output file
    output_file = joinpath(member_output_dir, "ocn_ts.nc")
    
    # Job script file
    script_file = joinpath(work_dir, "iter_$(iteration)", "member_$(member_id)_job.sh")
    
    # Extract parameters for this member from combined file
    # This will be done by reading line (member_id + 1) from param_file
    
    # Create SLURM job script
    script_content = """
    #!/bin/bash
    #SBATCH --job-name=climber_i$(iteration)_m$(member_id)
    #SBATCH --qos=long
    #SBATCH --time=200:00:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=64G
    #SBATCH --output=$(work_dir)/iter_$(iteration)/member_$(member_id)_%j.log
    #SBATCH --error=$(work_dir)/iter_$(iteration)/member_$(member_id)_%j.err
    
    echo "================================================================"
    echo "CLIMBER-X EKI Calibration Job"
    echo "================================================================"
    echo "Iteration: $(iteration)"
    echo "Member: $(member_id)"
    echo "Parameter file: $(param_file)"
    echo "Output directory: $(member_output_dir)"
    echo "Start time: \$(date)"
    echo "================================================================"
    
    # Change to CLIMBER-X directory
    cd $(CLIMBER_X_DIR)
    
    # Read parameters for this member (line member_id + 1)
    params=\$(sed -n '$((member_id + 1))p' $(param_file))
    
    # Parse the 6 parameters
    read -r diff_dia_min drag_topo_fac slope_max diff_iso diff_gm diff_dia_max <<< \$params
    
    echo "Parameters for member $(member_id):"
    echo "  ocn.diff_dia_min: \$diff_dia_min"
    echo "  ocn.drag_topo_fac: \$drag_topo_fac"
    echo "  ocn.slope_max: \$slope_max"
    echo "  ocn.diff_iso: \$diff_iso"
    echo "  ocn.diff_gm: \$diff_gm"
    echo "  ocn.diff_dia_max: \$diff_dia_max"
    echo ""
    
    # Run CLIMBER-X
    echo "Starting CLIMBER-X simulation..."
    ./runme -rs -q long -w 200:00:00 --omp 32 \\
        -o $(member_output_dir) \\
        -p ocn.diff_dia_min=\$diff_dia_min \\
           ocn.drag_topo_fac=\$drag_topo_fac \\
           ocn.slope_max=\$slope_max \\
           ocn.diff_iso=\$diff_iso \\
           ocn.diff_gm=\$diff_gm \\
           ocn.diff_dia_max=\$diff_dia_max \\
           ctl.nyears=$(CLIMBER_FIXED_PARAMS["ctl.nyears"]) \\
           ctl.co2_const=$(CLIMBER_FIXED_PARAMS["ctl.co2_const"]) \\
           ctl.fake_geo_const_file=$(CLIMBER_FIXED_PARAMS["ctl.fake_geo_const_file"]) \\
           ctl.fake_ice_const_file=$(CLIMBER_FIXED_PARAMS["ctl.fake_ice_const_file"]) \\
           ctl.restart_in_dir=$(CLIMBER_FIXED_PARAMS["ctl.restart_in_dir"]) \\
           ocn.l_noise_fw=$(CLIMBER_FIXED_PARAMS["ocn.l_noise_fw"])
    
    exit_code=\$?
    
    echo ""
    echo "================================================================"
    echo "Job finished"
    echo "End time: \$(date)"
    echo "Exit code: \$exit_code"
    
    # Check if output file exists
    if [ -f $(output_file) ] && [ \$exit_code -eq 0 ]; then
        echo "SUCCESS: CLIMBER-X run completed"
        echo "Output file: $(output_file)"
        file_size=\$(du -h $(output_file) | cut -f1)
        echo "File size: \$file_size"
        echo "SUCCESS" > $(dirname(output_file))/member_$(member_id).status
    else
        echo "FAILED: CLIMBER-X run failed"
        echo "Exit code: \$exit_code"
        echo "FAILED" > $(dirname(output_file))/member_$(member_id).status
        exit 1
    fi
    echo "================================================================"
    """
    
    open(script_file, "w") do f
        write(f, script_content)
    end
    
    return script_file, output_file
end

# ============================================
# OUTPUT VALIDATION
# ============================================

"""
Validate CLIMBER-X output file
"""
function validate_climber_output_file(output_file; min_size_bytes=1000000)
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
# JOB SUBMISSION AND COLLECTION
# ============================================

"""
Submit CLIMBER-X jobs for one iteration
"""
function submit_iteration_jobs_climber(params_i, iteration, work_dir, output_dir)
    N_ensemble = size(params_i, 2)
    job_trackers = JobTracker[]
    
    println("\n  Submitting $N_ensemble CLIMBER-X jobs for iteration $iteration...")
    println("  Expected runtime: ~200 hours per job")
    
    # Check disk space
    has_space, available_gb = check_disk_space(output_dir, min_gb_required=100, warn_gb=500)
    if !has_space
        error("Insufficient disk space")
    end
    
    # Write ONE combined parameter file for this iteration
    param_file = write_climber_params_for_iteration(params_i, iteration, work_dir)
    
    # Submit jobs - each reads its own line from the combined file
    for j in 1:N_ensemble
        script_file, output_file = create_climber_job_script(
            iteration, j, param_file, output_dir, work_dir
        )
        
        job_id = submit_job(script_file, max_retries=3)
        
        tracker = JobTracker(
            job_id,
            j,
            iteration,
            :submitted,
            now(),
            nothing,
            param_file,
            output_file
        )
        push!(job_trackers, tracker)
        
        if j % 10 == 0 || j == N_ensemble
            println("    Submitted $j/$N_ensemble jobs")
        end
        
        sleep(0.5)  # Rate limiting
    end
    
    println("  ✓ All $N_ensemble jobs submitted!")
    
    return job_trackers
end

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
            # Validate output file
            is_valid, msg = validate_climber_output_file(tracker.output_file)
            
            if is_valid
                try
                    # Process output and extract calibration statistics
                    calibration_stats, full_stats = process_climber_output(
                        tracker.output_file, pca_model,
                        remove_spinup=true, spinup_fraction=0.02
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

# ============================================
# MAIN CALIBRATION FUNCTION
# ============================================

function run_climber_x_calibration(;
    N_iterations=10,
    N_ensemble=50,
    output_dir="/p/tmp/karinako/eki_calibration",
    work_dir="/p/tmp/karinako/eki_calibration",
    check_interval_minutes=30,
    max_wait_days=10,
    pca_components=5)
    
    println("="^80)
    println("CLIMBER-X EKI CALIBRATION")
    println("="^80)
    println("Parameters: $(length(PARAM_NAMES)) ocean parameters")
    println("Observations: $pca_components PCA components + 2 dynamical statistics")
    println("Ensemble size: $N_ensemble")
    println("Iterations: $N_iterations")
    println("Output directory: $output_dir")
    println("="^80)
    
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
    
    # Setup prior - TRUE uniform distributions
    println("\nSetting up prior distributions...")
    using Distributions  # for Uniform
    
    prior_dists = []
    for name in PARAM_NAMES
        bounds = PRIOR_BOUNDS[name]
        # Create uniform distribution in physical space
        uniform_dist = Parameterized(Uniform(bounds[1], bounds[2]))
        # No constraint needed - distribution is already in physical space
        constraint = no_constraint()
        push!(prior_dists, ParameterDistribution(uniform_dist, constraint, name))
    end
    prior = combine_distributions(prior_dists)
    
    println("  ✓ Prior configured for $(length(PARAM_NAMES)) parameters")
    
    # Process default run to get observations
    println("\nProcessing default run for target observations...")
    println("  Default run: $DEFAULT_RUN_OUTPUT")
    
    if !isfile(DEFAULT_RUN_OUTPUT)
        error("Default run output not found: $DEFAULT_RUN_OUTPUT")
    end
    
    # We'll fit PCA after first iteration, so for now just read the default stats
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
    
    # Observation noise covariance
    obs_noise_cov = Diagonal(OBS_UNCERTAINTIES.^2)
    
    # Initialize EKI
    println("\nInitializing EKI process...")
    initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
    eks_process = Sampler(prior)
    
    # We'll set y_obs after PCA is fit
    # For now, create placeholder
    y_obs_placeholder = zeros(7)
    
    eksobj = EnsembleKalmanProcess(
        initial_ensemble,
        y_obs_placeholder,  # Will update after PCA
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
        "default_run" => DEFAULT_RUN_OUTPUT
    )
    
    save_checkpoint(0, eksobj, prior, param_history, 
                   y_obs_placeholder, obs_noise_cov, metadata, checkpoint_dir)
    
    # PCA model (will be fit after first iteration)
    pca_model = nothing
    y_obs = nothing
    
    # Main iteration loop
    for i in 1:N_iterations
        iter_start_time = now()
        
        println("\n" * "="^80)
        println("ITERATION $i/$N_iterations")
        println("="^80)
        
        params_i = get_ϕ_final(prior, eksobj)
        
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
        
        # After first iteration: fit PCA and set observations
        if i == 1 && isnothing(pca_model)
            println("\n  Fitting PCA model from iteration 1 ensemble...")
            
            # Collect output files that completed successfully
            ensemble_files = [tracker.output_file for tracker in job_trackers 
                            if tracker.status == :completed && 
                               validate_climber_output_file(tracker.output_file)[1]]
            
            pca_model, _ = initialize_pca_model(
                DEFAULT_RUN_OUTPUT, 
                ensemble_files;
                n_components=pca_components,
                remove_spinup=true,
                spinup_fraction=0.02
            )
            
            # Now we can extract target observations with PCA
            default_calibration_stats, _ = process_climber_output(
                DEFAULT_RUN_OUTPUT, pca_model,
                remove_spinup=true, spinup_fraction=0.02
            )
            
            y_obs = default_calibration_stats
            
            println("\n  Target observations (from default run):")
            println("    PCA components: $(round.(y_obs[1:5], digits=4))")
            println("    Avg waiting time: $(round(y_obs[6], digits=1)) years")
            println("    Avg stadial duration: $(round(y_obs[7], digits=1)) years")
            
            # Update EKI with correct observations
            eksobj = EnsembleKalmanProcess(
                get_ϕ(eksobj, 1),  # Use iteration 1 ensemble
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

println("\nStarting CLIMBER-X EKI calibration...")
println("Note: First iteration will be used to fit PCA model")
println("")

eksobj, param_history, metadata, pca_model = run_climber_x_calibration(
    N_iterations=10,
    N_ensemble=50,
    output_dir="/p/tmp/karinako/eki_calibration",
    work_dir="/p/tmp/karinako/eki_calibration",
    check_interval_minutes=30,
    max_wait_days=10,
    pca_components=5
)

println("\n✓ Calibration completed successfully!")
println("Results saved in: /p/tmp/karinako/eki_calibration/")
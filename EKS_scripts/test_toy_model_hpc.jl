using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using LinearAlgebra
using Statistics
using Random
using JLD2
using Dates
using Printf

# Include job management functions
include("eks_job_management.jl")

# ============================================
# TOY ENERGY BALANCE MODEL (2 parameters)
# ============================================

"""
Simple energy balance climate model
"""
function simple_climate_model(θ; n_years=100)
    climate_sensitivity = θ[1]
    time_constant = θ[2]
    
    T_eq = 15.0 + 2.0 * (climate_sensitivity - 3.0)
    
    times = 1:n_years
    temperatures = T_eq .* (1 .- exp.(-times ./ time_constant))
    
    Random.seed!(42)
    temperatures .+= 0.1 * randn(n_years)
    
    return temperatures
end

# ============================================
# PARAMETER FILE WRITING (YOUR FORMAT)
# ============================================

"""
Write all ensemble parameters to a single file (one line per member)
"""
function write_params_for_iteration(params_i, iteration, work_dir)
    """
    Creates one file with all ensemble members
    Format:
    param1_name param2_name
    value1_member1 value2_member1
    value1_member2 value2_member2
    ...
    """
    param_file = joinpath(work_dir, "iter_$(iteration)", "params_ensemble.txt")
    mkpath(dirname(param_file))
    
    # Parameter names
    param_names = [
        "climate_sensitivity",
        "time_constant"
    ]
    
    N_ensemble = size(params_i, 2)
    
    open(param_file, "w") do f
        # Write header with parameter names
        println(f, join(param_names, " "))
        
        # Write each ensemble member as a row
        for j in 1:N_ensemble
            println(f, join([@sprintf("%.17g", val) for val in params_i[:, j]], " "))
        end
    end
    
    return param_file
end

# ============================================
# TOY MODEL JOB SCRIPT
# ============================================

"""
Modified job script that reads one line from the combined file
"""
function create_toy_model_job_script(iteration, member_id, param_file, output_dir, work_dir; 
                                     simulate_runtime_seconds=60)
    output_file = joinpath(output_dir, "iter_$(iteration)", "member_$(member_id)_output.txt")
    mkpath(dirname(output_file))
    
    script_file = joinpath(work_dir, "iter_$(iteration)", "member_$(member_id)_job.sh")
    
    # Create a Julia script that will be run by the job
    model_script_file = joinpath(work_dir, "iter_$(iteration)", "member_$(member_id)_model.jl")
    
    model_script_content = """
    using Random
    using Statistics
    
    # Read parameters from combined file (specific line for this member)
    function read_params(param_file, member_id)
        lines = readlines(param_file)
        # Line 1 is header, line (member_id + 1) is this member's params
        value_line = lines[member_id + 1]
        return [parse(Float64, val) for val in split(value_line)]
    end
    
    # Simple climate model (2 parameters)
    function simple_climate_model(θ; n_years=100)
        climate_sensitivity = θ[1]
        time_constant = θ[2]
        
        T_eq = 15.0 + 2.0 * (climate_sensitivity - 3.0)
        times = 1:n_years
        temperatures = T_eq .* (1 .- exp.(-times ./ time_constant))
        
        Random.seed!(42)
        temperatures .+= 0.1 * randn(n_years)
        
        return temperatures
    end
    
    # Main execution
    param_file = "$(param_file)"
    member_id = $(member_id)
    output_file = "$(output_file)"
    
    println("Reading parameters from: \$param_file (line \$(member_id + 1))")
    θ = read_params(param_file, member_id)
    println("Parameters for member \$member_id: \$θ")
    
    println("Running climate model...")
    sleep($(simulate_runtime_seconds))
    
    temps = simple_climate_model(θ)
    
    println("Computing statistics...")
    n = length(temps)
    times = 1:n
    trend = (n * sum(times .* temps) - sum(times) * sum(temps)) / 
            (n * sum(times.^2) - sum(times)^2)
    
    stats = [
        mean(temps),
        temps[end],
        trend,
        std(temps)
    ]
    
    println("Saving results to: \$output_file")
    open(output_file, "w") do f
        println(f, "# Climate model output statistics")
        println(f, "# Member: $(member_id), Iteration: $(iteration)")
        println(f, "mean_temp \$(stats[1])")
        println(f, "final_temp \$(stats[2])")
        println(f, "trend \$(stats[3])")
        println(f, "variability \$(stats[4])")
    end
    
    println("SUCCESS")
    """
    
    open(model_script_file, "w") do f
        write(f, model_script_content)
    end
    
    # Create SLURM job script
    script_content = """
    #!/bin/bash
    #SBATCH --job-name=eks_toy_i$(iteration)_m$(member_id)
    #SBATCH --qos=short              # ADD THIS LINE
    #SBATCH --time=00:10:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --mem=2G
    #SBATCH --output=$(work_dir)/iter_$(iteration)/member_$(member_id)_%j.log
    #SBATCH --error=$(work_dir)/iter_$(iteration)/member_$(member_id)_%j.err
    
    # Load Julia module (adjust for your HPC)
    # module load julia/1.9
    
    echo "Starting toy model job"
    echo "Iteration: $(iteration), Member: $(member_id)"
    echo "Parameters file: $(param_file)"
    echo "Reading line $(member_id + 1) from parameter file"
    echo "Output file: $(output_file)"
    date
    
    # Run the Julia model script
    julia $(model_script_file)
    
    exit_code=\$?
    
    if [ -f $(output_file) ] && [ \$exit_code -eq 0 ]; then
        echo "SUCCESS: Model run completed"
        echo "SUCCESS" > $(dirname(output_file))/member_$(member_id).status
    else
        echo "FAILED: Model run failed with exit code \$exit_code"
        echo "FAILED" > $(dirname(output_file))/member_$(member_id).status
        exit 1
    fi
    
    date
    """
    
    open(script_file, "w") do f
        write(f, script_content)
    end
    
    return script_file, output_file
end

# ============================================
# OUTPUT VALIDATION AND READING
# ============================================

"""
Validate toy model output file
"""
function validate_toy_output_file(output_file; min_size_bytes=100)
    if !isfile(output_file)
        return false, "File does not exist"
    end
    
    file_size = filesize(output_file)
    if file_size < min_size_bytes
        return false, "File too small: $(file_size) bytes"
    end
    
    try
        lines = readlines(output_file)
        if length(lines) < 3
            return false, "File has too few lines"
        end
        
        return true, "Valid"
    catch e
        return false, "Cannot read file: $e"
    end
end

"""
Compute summary statistics from toy model output
"""
function compute_toy_summary_statistics(output_file)
    is_valid, msg = validate_toy_output_file(output_file)
    
    if !is_valid
        error("Invalid output file: $msg")
    end
    
    try
        lines = readlines(output_file)
        
        stats = Float64[]
        
        for line in lines
            if startswith(line, "#")
                continue
            end
            
            parts = split(line)
            if length(parts) == 2
                value = parse(Float64, parts[2])
                push!(stats, value)
            end
        end
        
        if length(stats) != 4
            error("Expected 4 statistics, got $(length(stats))")
        end
        
        if any(isnan.(stats)) || any(isinf.(stats))
            error("Statistics contain NaN or Inf values")
        end
        
        return stats
        
    catch e
        @error "Failed to compute statistics from $output_file" exception=e
        rethrow(e)
    end
end

# ============================================
# JOB SUBMISSION AND COLLECTION
# ============================================

"""
Submit jobs using single combined parameter file per iteration
"""
function submit_iteration_jobs_toy(params_i, iteration, work_dir, output_dir; 
                                   simulate_runtime_seconds=60)
    N_ensemble = size(params_i, 2)
    job_trackers = JobTracker[]
    
    println("\n  Submitting $N_ensemble TOY MODEL jobs for iteration $iteration...")
    println("  (Each job will run for ~$simulate_runtime_seconds seconds)")
    
    has_space, available_gb = check_disk_space(output_dir, min_gb_required=1, warn_gb=5)
    if !has_space
        error("Insufficient disk space")
    end
    
    # Write ONE combined parameter file for this iteration
    param_file = write_params_for_iteration(params_i, iteration, work_dir)
    println("  ✓ Wrote combined parameter file: $param_file")
    
    # Submit jobs - each reads its own line from the combined file
    for j in 1:N_ensemble
        script_file, output_file = create_toy_model_job_script(
            iteration, j, param_file, output_dir, work_dir,
            simulate_runtime_seconds=simulate_runtime_seconds
        )
        
        job_id = submit_job(script_file, max_retries=3)
        
        tracker = JobTracker(
            job_id,
            j,
            iteration,
            :submitted,
            now(),
            nothing,
            param_file,  # All members share the same param file
            output_file
        )
        push!(job_trackers, tracker)
        
        if j % 5 == 0 || j == N_ensemble
            println("    Submitted $j/$N_ensemble jobs")
        end
        
        sleep(0.1)
    end
    
    println("  ✓ All $N_ensemble jobs submitted!")
    
    return job_trackers
end

"""
Collect results from toy model
"""
function collect_toy_iteration_results(job_trackers, y_obs; max_failures_allowed=2)
    N_ensemble = length(job_trackers)
    n_outputs = length(y_obs)
    G_ensemble = zeros(n_outputs, N_ensemble)
    
    n_failures = 0
    
    println("\n  Collecting TOY MODEL results from $N_ensemble jobs...")
    
    for (j, tracker) in enumerate(job_trackers)
        if tracker.status == :completed
            try
                G_ensemble[:, j] = compute_toy_summary_statistics(tracker.output_file)
                
                if j % 5 == 0
                    println("    Processed $j/$N_ensemble outputs")
                end
                
            catch e
                @warn "Failed to process member $(tracker.member_id): $e"
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
        @error "Too many failures: $n_failures/$N_ensemble"
        error("Iteration failed")
    elseif n_failures > 0
        @warn "$n_failures/$N_ensemble members failed"
    end
    
    println("  ✓ Results collected: $(N_ensemble - n_failures) successful")
    
    return G_ensemble
end

# ============================================
# MAIN TEST CALIBRATION
# ============================================

function run_toy_model_test(;
    N_iterations=5,
    N_ensemble=50,
    output_dir="eks_toy_test",
    work_dir="eks_toy_work",
    check_interval_minutes=1,
    max_wait_days=1,
    simulate_runtime_seconds=60)
    
    println("="^60)
    println("EKS TOY MODEL TEST ON HPC (2 Parameters)")
    println("="^60)
    println("This test uses the simple energy balance model")
    println("Parameters: climate_sensitivity, time_constant")
    println("Each 'model run' will take ~$simulate_runtime_seconds seconds")
    println("Total expected time: ~$(N_iterations * simulate_runtime_seconds / 60) minutes")
    println("="^60)
    
    mkpath(output_dir)
    mkpath(work_dir)
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    mkpath(checkpoint_dir)
    
    has_space, _ = check_disk_space(output_dir, min_gb_required=1, warn_gb=5)
    if !has_space
        error("Insufficient disk space")
    end
    
    # Setup prior - 2 parameters
    prior = combine_distributions([
        constrained_gaussian("climate_sensitivity", 4.0, 1.5, 1.0, 6.0),
        constrained_gaussian("time_constant", 30.0, 10.0, 5.0, 50.0)
    ])
    
    # Generate "truth"
    θ_true = [3.0, 20.0]
    println("\nTrue parameters: $θ_true")
    
    temps_true = simple_climate_model(θ_true)
    
    n = length(temps_true)
    times = 1:n
    trend = (n * sum(times .* temps_true) - sum(times) * sum(temps_true)) / 
            (n * sum(times.^2) - sum(times)^2)
    
    y_true = [
        mean(temps_true),
        temps_true[end],
        trend,
        std(temps_true)
    ]
    
    # Add observation noise
    obs_noise_std = [0.2, 0.3, 0.01, 0.05]
    Random.seed!(123)
    y_obs = y_true .+ obs_noise_std .* randn(length(y_true))
    obs_noise_cov = Diagonal(obs_noise_std.^2)
    
    println("Observations: $y_obs")
    
    # Initialize EKS
    initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
    eks_process = Sampler(prior)
    eksobj = EnsembleKalmanProcess(
        initial_ensemble,
        y_obs,
        obs_noise_cov,
        eks_process,
        verbose=true
    )
    
    param_history = zeros(2, N_iterations + 1, N_ensemble)
    param_history[:, 1, :] = get_ϕ_final(prior, eksobj)
    
    metadata = Dict(
        "start_time" => now(),
        "N_iterations" => N_iterations,
        "N_ensemble" => N_ensemble,
        "θ_true" => θ_true,
        "y_obs" => y_obs
    )
    
    save_checkpoint(0, eksobj, prior, param_history, 
                   y_obs, obs_noise_cov, metadata, checkpoint_dir)
    
    # Main iteration loop
    for i in 1:N_iterations
        iter_start_time = now()
        
        println("\n" * "="^60)
        println("ITERATION $i/$N_iterations")
        println("="^60)
        
        params_i = get_ϕ_final(prior, eksobj)
        
        job_trackers = submit_iteration_jobs_toy(
            params_i, i, work_dir, output_dir,
            simulate_runtime_seconds=simulate_runtime_seconds
        )
        
        save_job_trackers(job_trackers, i, output_dir)
        
        result = wait_for_iteration_completion(
            job_trackers;
            check_interval_minutes=check_interval_minutes,
            max_wait_days=max_wait_days,
            output_dir=output_dir
        )
        
        if result == :timeout
            error("Iteration $i timed out")
        end
        
        G_ensemble = collect_toy_iteration_results(job_trackers, y_obs, max_failures_allowed=2)
        
        println("\n  Updating ensemble...")
        update_ensemble!(eksobj, G_ensemble)
        param_history[:, i+1, :] = get_ϕ_final(prior, eksobj)
        
        current_mean = get_ϕ_mean_final(prior, eksobj)
        current_std = std(get_ϕ_final(prior, eksobj), dims=2)
        
        println("\n  Parameter estimates:")
        println("    Climate sensitivity: $(round(current_mean[1], digits=3)) ± $(round(current_std[1], digits=3)) (true: $(θ_true[1]))")
        println("    Time constant:       $(round(current_mean[2], digits=3)) ± $(round(current_std[2], digits=3)) (true: $(θ_true[2]))")
        
        iter_duration = now() - iter_start_time
        println("\n  Iteration duration: $(iter_duration)")
        
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
    
    println("\n" * "="^60)
    println("TOY MODEL TEST COMPLETE!")
    println("="^60)
    
    println("\nFinal results:")
    println("Parameter            | True    | Estimated | Std     | Error")
    println("-"^60)
    @printf("%-20s | %7.3f | %9.3f | %7.3f | %+6.3f\n", 
            "Climate sensitivity", θ_true[1], θ_optimal[1], θ_std[1], θ_optimal[1] - θ_true[1])
    @printf("%-20s | %7.3f | %9.3f | %7.3f | %+6.3f\n", 
            "Time constant", θ_true[2], θ_optimal[2], θ_std[2], θ_optimal[2] - θ_true[2])
    
    return eksobj, param_history, metadata
end

# ============================================
# RUN THE TEST
# ============================================

eksobj, param_history, metadata = run_toy_model_test(
    N_iterations=3,
    N_ensemble=50,
    output_dir="eks_toy_test",
    work_dir="eks_toy_work",
    check_interval_minutes=1,
    max_wait_days=1,
    simulate_runtime_seconds=60
)

println("\n✓ Test completed successfully!")
println("Check output in eks_toy_test/ directory")
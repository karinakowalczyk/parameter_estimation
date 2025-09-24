#!/bin/bash

# Main ensemble directory
ENSEMBLE_DIR="/p/tmp/karinako/ocean_ppe_300"
mkdir -p "$ENSEMBLE_DIR"

# Create info.txt file with header
echo "runid ocn.diff_dia_min ocn.drag_topo_fac ocn.slope_max ocn.diff_iso ocn.diff_gm ocn.diff_dia_max rundir" > "$ENSEMBLE_DIR/info.txt"

# Create params.txt file with header
echo "ocn.diff_dia_min ocn.drag_topo_fac ocn.slope_max ocn.diff_iso ocn.diff_gm ocn.diff_dia_max ctl.nyears ctl.co2_const ctl.fake_geo_const_file ctl.fake_ice_const_file ctl.restart_in_dir ocn.l_noise_fw" > "$ENSEMBLE_DIR/params.txt"

# Read the LHS file and submit jobs
runid=1
tail -n +2 ocean_params_300.txt | while IFS= read -r line; do
    # Parse the six parameters from the line
    read -r diff_dia_min drag_topo_fac slope_max diff_iso diff_gm diff_dia_max <<< $(echo "$line" | awk '{print $1, $2, $3, $4, $5, $6}')

    # Run directory
    run_dir="$ENSEMBLE_DIR/$runid"

    # Add to info.txt
    printf "%5d %15s %15s %12s %10s %10s %15s %s\n" \
        "$runid" "$diff_dia_min" "$drag_topo_fac" "$slope_max" "$diff_iso" "$diff_gm" "$diff_dia_max" "$runid" >> "$ENSEMBLE_DIR/info.txt"

    # Add to params.txt
    echo "$diff_dia_min $drag_topo_fac $slope_max $diff_iso $diff_gm $diff_dia_max 10000 190 input/geo_ice_tarasov_12ka.nc input/geo_ice_tarasov_12ka.nc /home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000 T" >> "$ENSEMBLE_DIR/params.txt"

    # Submit job with corrected single -p block
    echo "Submitting job $runid to queue"
    ./runme -rs -q standby -w 24:00:00 --omp 32 \
        -o "$run_dir" \
        -p ocn.diff_dia_min="$diff_dia_min" \
           ocn.drag_topo_fac="$drag_topo_fac" \
           ocn.slope_max="$slope_max" \
           ocn.diff_iso="$diff_iso" \
           ocn.diff_gm="$diff_gm" \
           ocn.diff_dia_max="$diff_dia_max" \
           ctl.nyears=10000 \
           ctl.co2_const=190 \
           ctl.fake_geo_const_file=input/geo_ice_tarasov_12ka.nc \
           ctl.fake_ice_const_file=input/geo_ice_tarasov_12ka.nc \
           ctl.restart_in_dir="/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000" \
           ocn.l_noise_fw=T &

    # Rate limiting: wait a bit between submissions
    sleep 1

    runid=$((runid+1))
done

echo "All jobs submitted to SLURM queue via runme!"
echo "Ensemble directory: $ENSEMBLE_DIR"
echo "Run directories: $ENSEMBLE_DIR/1 ... $ENSEMBLE_DIR/N"
echo "Info file: $ENSEMBLE_DIR/info.txt"
echo "Params file: $ENSEMBLE_DIR/params.txt"
echo "Check job status with: squeue -u \$USER"

tasks="design_bench soo_bench co bboplace_bench bbob real_world hpob"
# tasks="soo_bench co bboplace_bench bbob real_world"
ckpt_path="logs/baseline_embed_regress_proj_t5/runs/2025-01-13_23-54-36_seed42/checkpoints/last.ckpt"

MAX_JOBS=6
AVAILABLE_GPUS="3"
MAX_RETRIES=0

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

for task in $tasks; do 
    for seed in {42..49}; do
        check_jobs
        gpu_allocation=$(get_gpu_allocation $job_number)
        ((job_number++))
        run_with_retry "src/train_embed_regressor.py \
            experiment=embed_regress_proj_t5 \
            ++seed=${seed} \
            ++train=false \
            ++test_suites=${task} \
            ++ckpt_path=${ckpt_path}" \
            "$gpu_allocation" & 
    done
done 

wait
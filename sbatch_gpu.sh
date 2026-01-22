#!/bin/sh

#SBATCH --job-name=stream_bc    # The job name.
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --time=10-00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/ivi/ilps/personal/gpoerwa/test-torch-ilps/output/%x-%A-%a.log
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#load the conda test env
source ~/.bashrc
conda activate test
echo "activate the conda env"

# Set up distributed training environment variables, in case of running job array
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# # Find a free port dynamically
# find_free_port() {
#     python3 -c "
# import socket
# import random
# for _ in range(100):
#     port = random.randint(49152, 65535)
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             s.bind(('', port))
#             s.listen(1)
#             print(port)
#             break
#     except OSError:
#         continue
# else:
#     print(29500)
# "
# }


# export MASTER_PORT=$(find_free_port)

# Set proper environment variables
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Set up API keys, wandb for job monitoring 
export WANDB_API_KEY=<WANDB_API_KEY> \
export HF_HOME="/ivi/ilps/personal/gpoerwa/.cache/huggingface" \
export HF_DATASETS_CACHE="/ivi/ilps/scratch/ct" \
export HF_KEY=<HF_API_KEY> \
export TRANSFORMERS_CACHE="/ivi/ilps/personal/gpoerwa/.cache/huggingface/models" \
export TORCH_HOME="/ivi/ilps/personal/gpoerwa/.cache/torch" \
export WANDB_CACHE_DIR="/ivi/ilps/personal/gpoerwa/.cache/wandb" \
export PYTORCH_DISABLE_TORCH_LOAD_CHECK=1

# Add current directory to Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Record start time
start_time=$(date +%s)

#wandb disabled
python bert_similarity.py
#enabled
python bert_similarity.py --wandb_logging True

# Check the exit code
exit_code=$?

echo "Stream $SLURM_ARRAY_TASK_ID finished with exit code: $exit_code"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"

# Exit with the same code as the training script
exit $exit_code



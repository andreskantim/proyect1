#!/bin/bash
# Script to monitor SLURM job progress

if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    echo ""
    echo "Monitor a running SLURM job and show training progress"
    exit 1
fi

JOB_ID=$1

echo "================================================"
echo "Monitoring Job $JOB_ID"
echo "================================================"

# Check job status
echo ""
echo "Job Status:"
squeue -j $JOB_ID -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

echo ""
echo "Job Details:"
scontrol show job $JOB_ID | grep -E "JobId|JobName|UserId|Partition|NodeList|State|RunTime|TimeLimit|WorkDir|StdOut|StdErr"

# Show recent output
echo ""
echo "================================================"
echo "Recent Output (last 50 lines):"
echo "================================================"

LOG_FILE="logs/train_${JOB_ID}.out"

if [ -f "$LOG_FILE" ]; then
    tail -n 50 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi

# Show GPU usage on the node
echo ""
echo "================================================"
echo "GPU Usage:"
echo "================================================"

NODE=$(squeue -j $JOB_ID -h -o "%N")
if [ ! -z "$NODE" ]; then
    srun --jobid=$JOB_ID nvidia-smi
fi

echo ""
echo "================================================"
echo "To cancel job: scancel $JOB_ID"
echo "To follow live output: tail -f $LOG_FILE"
echo "================================================"

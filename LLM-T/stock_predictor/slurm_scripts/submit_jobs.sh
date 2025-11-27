#!/bin/bash
# Helper script to submit training jobs

echo "================================================"
echo "Market GPT - Job Submission Helper"
echo "================================================"

echo ""
echo "Available jobs:"
echo "  1) Train on Bitcoin (initial training)"
echo "  2) Test on Gold (walk-forward)"
echo "  3) Both (train Bitcoin, then test Gold)"
echo "  4) Quick test (small data for testing)"
echo ""

read -p "Select job to submit (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Submitting Bitcoin training job..."
        JOB_ID=$(sbatch slurm_scripts/train_bitcoin_a100.sh | awk '{print $4}')
        echo "Job submitted with ID: $JOB_ID"
        echo ""
        echo "Monitor with: bash slurm_scripts/monitor_job.sh $JOB_ID"
        echo "Cancel with: scancel $JOB_ID"
        ;;

    2)
        echo ""
        echo "Submitting Gold testing job..."
        JOB_ID=$(sbatch slurm_scripts/test_gold_a100.sh | awk '{print $4}')
        echo "Job submitted with ID: $JOB_ID"
        echo ""
        echo "Monitor with: bash slurm_scripts/monitor_job.sh $JOB_ID"
        echo "Cancel with: scancel $JOB_ID"
        ;;

    3)
        echo ""
        echo "Submitting Bitcoin training job..."
        JOB1=$(sbatch slurm_scripts/train_bitcoin_a100.sh | awk '{print $4}')
        echo "Bitcoin job submitted with ID: $JOB1"

        echo ""
        echo "Submitting Gold testing job (depends on Bitcoin job)..."
        JOB2=$(sbatch --dependency=afterok:$JOB1 slurm_scripts/test_gold_a100.sh | awk '{print $4}')
        echo "Gold job submitted with ID: $JOB2"
        echo ""
        echo "Jobs submitted in sequence: $JOB1 -> $JOB2"
        echo "Monitor Bitcoin: bash slurm_scripts/monitor_job.sh $JOB1"
        echo "Monitor Gold: bash slurm_scripts/monitor_job.sh $JOB2"
        ;;

    4)
        echo ""
        echo "Running quick test locally (no SLURM)..."
        python train_bitcoin.py \
            --config configs/quick_test.json \
            --output_dir checkpoints/quick_test \
            --log_dir logs/quick_test
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Job submission complete!"
echo "================================================"
echo ""
echo "Useful commands:"
echo "  squeue -u \$USER           - Show your jobs"
echo "  squeue -j <job_id>        - Show specific job"
echo "  scancel <job_id>          - Cancel job"
echo "  scontrol show job <id>    - Job details"
echo ""

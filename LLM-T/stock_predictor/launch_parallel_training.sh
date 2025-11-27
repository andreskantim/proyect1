#!/bin/bash

# =============================================================================
# Parallel Training Launcher for Case A and Case B
# =============================================================================
# This script launches both Case A (600 assets) and Case B (100 assets)
# training jobs in parallel on separate GPU nodes.
#
# Requirements:
#   - Case A: 2×A100 GPUs, 64 CPUs, 128GB RAM
#   - Case B: 2×A100 GPUs, 64 CPUs, 128GB RAM
#   - Total: 4 GPUs across 2 nodes
# =============================================================================

set -e

echo "======================================================================"
echo "Parallel Multi-Market Training Launcher"
echo "======================================================================"
echo "Start time: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor"
CASE_A_SCRIPT="${PROJECT_ROOT}/case_a_full_market/slurm_scripts/train_full_a100.sh"
CASE_B_SCRIPT="${PROJECT_ROOT}/case_b_reduced/slurm_scripts/train_reduced_a100.sh"

# Verify scripts exist
echo "🔍 Verifying training scripts..."
if [ ! -f "${CASE_A_SCRIPT}" ]; then
    echo -e "${RED}❌ Case A script not found: ${CASE_A_SCRIPT}${NC}"
    exit 1
fi

if [ ! -f "${CASE_B_SCRIPT}" ]; then
    echo -e "${RED}❌ Case B script not found: ${CASE_B_SCRIPT}${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Both scripts found${NC}"
echo ""

# Check SLURM availability
echo "🖥️  Checking cluster resources..."
if ! command -v squeue &> /dev/null; then
    echo -e "${RED}❌ SLURM not found. Are you on the cluster?${NC}"
    exit 1
fi

# Show current queue
echo ""
echo "Current SLURM queue:"
squeue -u $(whoami) 2>/dev/null || echo "No jobs currently running"
echo ""

# Check available A100 GPUs
echo "📊 Checking A100 GPU availability..."
AVAILABLE_GPUS=$(sinfo -p long -o "%D %G" | grep a100 | head -1 | awk '{print $2}' | sed 's/gpu:a100://')
echo "Available A100 GPUs in 'long' partition: ${AVAILABLE_GPUS:-unknown}"
echo ""

# Ask for confirmation
echo "======================================================================"
echo "⚠️  You are about to launch TWO training jobs:"
echo ""
echo -e "${BLUE}📦 Case A: Full Market (600 assets)${NC}"
echo "   - 2×A100 GPUs, 64 CPUs, 128GB RAM"
echo "   - Estimated time: 7-10 days"
echo "   - Model size: ~85M parameters"
echo ""
echo -e "${BLUE}📦 Case B: Reduced Market (100 assets)${NC}"
echo "   - 2×A100 GPUs, 64 CPUs, 128GB RAM"
echo "   - Estimated time: 3-5 days"
echo "   - Model size: ~45M parameters"
echo ""
echo -e "${YELLOW}Total resources: 4 GPUs, 128 CPUs, 256GB RAM${NC}"
echo "======================================================================"
echo ""

read -p "Do you want to proceed? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Launch cancelled by user"
    exit 0
fi

# Launch Case A
echo "======================================================================"
echo "🚀 Launching Case A: Full Market (600 assets)"
echo "======================================================================"
JOB_A=$(sbatch --parsable "${CASE_A_SCRIPT}")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Case A submitted successfully!${NC}"
    echo "   Job ID: ${JOB_A}"
else
    echo -e "${RED}❌ Failed to submit Case A${NC}"
    exit 1
fi

echo ""
sleep 2

# Launch Case B
echo "======================================================================"
echo "🚀 Launching Case B: Reduced Market (100 assets)"
echo "======================================================================"
JOB_B=$(sbatch --parsable "${CASE_B_SCRIPT}")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Case B submitted successfully!${NC}"
    echo "   Job ID: ${JOB_B}"
else
    echo -e "${RED}❌ Failed to submit Case B${NC}"
    echo -e "${YELLOW}⚠️  Case A is still running (Job ID: ${JOB_A})${NC}"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ Both jobs submitted successfully!"
echo "======================================================================"
echo ""
echo "Job IDs:"
echo "  Case A (600 assets): ${JOB_A}"
echo "  Case B (100 assets): ${JOB_B}"
echo ""
echo "📊 Monitor jobs with:"
echo "  squeue -j ${JOB_A},${JOB_B}"
echo ""
echo "📋 View logs:"
echo "  Case A: tail -f ${PROJECT_ROOT}/case_a_full_market/logs/full_${JOB_A}.out"
echo "  Case B: tail -f ${PROJECT_ROOT}/case_b_reduced/logs/reduced_${JOB_B}.out"
echo ""
echo "❌ Cancel jobs:"
echo "  scancel ${JOB_A}  # Cancel Case A"
echo "  scancel ${JOB_B}  # Cancel Case B"
echo "  scancel ${JOB_A} ${JOB_B}  # Cancel both"
echo ""

# Save job info
JOB_INFO_FILE="${PROJECT_ROOT}/parallel_training_jobs.txt"
{
    echo "======================================================================"
    echo "Parallel Training Jobs"
    echo "======================================================================"
    echo "Launch time: $(date)"
    echo ""
    echo "Case A (600 assets):"
    echo "  Job ID: ${JOB_A}"
    echo "  Script: ${CASE_A_SCRIPT}"
    echo "  Output: ${PROJECT_ROOT}/case_a_full_market/logs/full_${JOB_A}.out"
    echo "  Error:  ${PROJECT_ROOT}/case_a_full_market/logs/full_${JOB_A}.err"
    echo ""
    echo "Case B (100 assets):"
    echo "  Job ID: ${JOB_B}"
    echo "  Script: ${CASE_B_SCRIPT}"
    echo "  Output: ${PROJECT_ROOT}/case_b_reduced/logs/reduced_${JOB_B}.out"
    echo "  Error:  ${PROJECT_ROOT}/case_b_reduced/logs/reduced_${JOB_B}.err"
    echo ""
    echo "Monitor: squeue -j ${JOB_A},${JOB_B}"
    echo "Cancel:  scancel ${JOB_A} ${JOB_B}"
    echo "======================================================================"
} > "${JOB_INFO_FILE}"

echo "💾 Job information saved to: ${JOB_INFO_FILE}"
echo ""

# Create monitoring script
MONITOR_SCRIPT="${PROJECT_ROOT}/monitor_training.sh"
cat > "${MONITOR_SCRIPT}" << 'EOF'
#!/bin/bash

# Load job IDs from saved file
PROJECT_ROOT="/mnt/netapp2/Home_FT2/home/ulc/cursos/curso396/LLM-T/stock_predictor"
JOB_INFO_FILE="${PROJECT_ROOT}/parallel_training_jobs.txt"

if [ ! -f "${JOB_INFO_FILE}" ]; then
    echo "❌ Job info file not found: ${JOB_INFO_FILE}"
    echo "   Run launch_parallel_training.sh first"
    exit 1
fi

# Extract job IDs
JOB_A=$(grep "Case A" -A 1 "${JOB_INFO_FILE}" | grep "Job ID" | awk '{print $3}')
JOB_B=$(grep "Case B" -A 1 "${JOB_INFO_FILE}" | grep "Job ID" | awk '{print $3}')

if [ -z "${JOB_A}" ] || [ -z "${JOB_B}" ]; then
    echo "❌ Could not extract job IDs from ${JOB_INFO_FILE}"
    exit 1
fi

echo "======================================================================"
echo "Training Progress Monitor"
echo "======================================================================"
echo "Monitoring jobs: ${JOB_A} (Case A), ${JOB_B} (Case B)"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "======================================================================"
    echo "Training Progress Monitor - $(date)"
    echo "======================================================================"
    echo ""

    # SLURM job status
    echo "📊 SLURM Job Status:"
    squeue -j ${JOB_A},${JOB_B} -o "%.10i %.12j %.10u %.8T %.10M %.6D %.20R" 2>/dev/null || echo "No jobs found (may have completed)"
    echo ""

    # Case A progress
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Case A: Full Market (600 assets) - Job ${JOB_A}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LOG_A="${PROJECT_ROOT}/case_a_full_market/logs/full_${JOB_A}.out"
    if [ -f "${LOG_A}" ]; then
        echo "Recent output:"
        tail -n 15 "${LOG_A}" | sed 's/^/  /'
    else
        echo "  ⏳ Log file not yet available"
    fi
    echo ""

    # Case B progress
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Case B: Reduced Market (100 assets) - Job ${JOB_B}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    LOG_B="${PROJECT_ROOT}/case_b_reduced/logs/reduced_${JOB_B}.out"
    if [ -f "${LOG_B}" ]; then
        echo "Recent output:"
        tail -n 15 "${LOG_B}" | sed 's/^/  /'
    else
        echo "  ⏳ Log file not yet available"
    fi
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Refreshing in 30 seconds... (Ctrl+C to exit)"

    sleep 30
done
EOF

chmod +x "${MONITOR_SCRIPT}"

echo "📈 Created monitoring script: ${MONITOR_SCRIPT}"
echo "   Run: ./monitor_training.sh"
echo ""

# Optionally start monitoring
read -p "Do you want to start the monitoring script now? (yes/no): " -r
echo ""

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    exec "${MONITOR_SCRIPT}"
else
    echo "You can start monitoring later by running:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  ./monitor_training.sh"
fi

echo ""
echo "======================================================================"
echo "✅ Setup complete!"
echo "======================================================================"

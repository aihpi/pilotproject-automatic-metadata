#!/bin/bash

# ==============================
# SLURM Job Configuration for Document Downloads
# ==============================

#SBATCH --job-name=download_docs
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition
#SBATCH --qos=aisc                          # QOS for AISC
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=16                  # 16 CPU cores (default workers)
#SBATCH --mem=64G                           # 64GB memory for download processing
#SBATCH --time=24:00:00                     # 24 hours
#SBATCH --constraint=ARCH:X86               # X86 architecture constraint
#SBATCH --output=logs/download_docs_%j.out
#SBATCH --error=logs/download_docs_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory if it doesn't exist
mkdir -p logs

# Print job information
echo "===== Document Download Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time Limit: ${SLURM_JOB_TIME_LIMIT}"
echo "Start Time: $(date)"
echo "=============================================="

# Load environment variables if .env.local exists
if [ -f ".env.local" ]; then
    echo "Loading environment variables from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
fi

# Set default project root if not defined
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(realpath "$0")")/../}"

# Navigate to project directory
echo "Navigating to project directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "Available CPU cores: $(nproc)"

# Set environment variables for optimization
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered

# ==============================
# Configuration with Environment Variables and Command Line Arguments
# ==============================

# Set default values if environment variables are not provided
XML_DIR="${XML_DIR:-data/xml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/documents}"
WORKERS="${WORKERS:-16}"                   # Default 16 for SBATCH
RETRIES="${RETRIES:-3}"
DELAY="${DELAY:-0.5}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DRY_RUN="${DRY_RUN:-false}"               # Optional: dry run mode

# Parse command line arguments and override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --xml-dir)
            XML_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --retries)
            RETRIES="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Available arguments:"
            echo "  --xml-dir XML_DIR           Directory containing XML files (default: data/xml)"
            echo "  --output-dir OUTPUT_DIR     Output directory for documents (default: data/documents)"
            echo "  --workers WORKERS           Number of parallel workers (default: 16)"
            echo "  --retries RETRIES           Maximum download retries (default: 3)"
            echo "  --delay DELAY               Delay between downloads in seconds (default: 0.5)"
            echo "  --log-level LOG_LEVEL       Logging level (default: INFO)"
            echo "  --dry-run                   Show what would be downloaded without downloading"
            exit 1
            ;;
    esac
done

# Validate workers doesn't exceed available CPUs
MAX_WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}
if [ "$WORKERS" -gt "$MAX_WORKERS" ]; then
    echo "WARNING: WORKERS ($WORKERS) exceeds available CPUs ($MAX_WORKERS)"
    echo "Setting WORKERS to $MAX_WORKERS"
    WORKERS=$MAX_WORKERS
fi

# ==============================
# Run Document Download
# ==============================

echo "Starting document download processing..."
echo "Configuration:"
echo "  XML directory: ${XML_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Workers: ${WORKERS}"
echo "  Max retries: ${RETRIES}"
echo "  Delay: ${DELAY} seconds"
echo "  Log level: ${LOG_LEVEL}"
echo "  Dry run: ${DRY_RUN}"
echo "=============================================="

# Build command arguments
CMD_ARGS=(
    --xml-dir "$XML_DIR"
    --output-dir "$OUTPUT_DIR"
    --workers "$WORKERS"
    --retries "$RETRIES"
    --delay "$DELAY"
    --log-level "$LOG_LEVEL"
)

# Add dry run if specified
if [ "$DRY_RUN" = "true" ]; then
    CMD_ARGS+=(--dry-run)
fi

# Display full command
echo "Command: python scripts/download_documents.py ${CMD_ARGS[*]}"
echo "=============================================="

# Record start time for performance tracking
START_TIME=$(date +%s)

# Execute the command
python scripts/download_documents.py "${CMD_ARGS[@]}"
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=============================================="
echo "Processing completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Document download completed successfully!"
    
    # Show output directory statistics if not dry run
    if [ "$DRY_RUN" != "true" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "Download statistics:"
        echo "  📁 Output directory: $OUTPUT_DIR"
        echo "     Size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "N/A")"
        
        # Count files by type
        PDF_COUNT=$(find "$OUTPUT_DIR" -name "*.pdf" -type f 2>/dev/null | wc -l)
        DOCX_COUNT=$(find "$OUTPUT_DIR" -name "*.docx" -type f 2>/dev/null | wc -l)
        DOC_COUNT=$(find "$OUTPUT_DIR" -name "*.doc" -type f 2>/dev/null | wc -l)
        HTML_COUNT=$(find "$OUTPUT_DIR" -name "*.html" -o -name "*.htm" -type f 2>/dev/null | wc -l)
        TOTAL_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)
        
        echo "     Files: $TOTAL_COUNT total"
        echo "       - PDF: $PDF_COUNT"
        echo "       - DOCX: $DOCX_COUNT"
        echo "       - DOC: $DOC_COUNT"
        echo "       - HTML: $HTML_COUNT"
    fi
    
elif [ $EXIT_CODE -eq 2 ]; then
    echo "⚠️  Document download completed with some failures!"
    echo "Check the logs for details about failed downloads."
    
else
    echo "❌ Document download failed with exit code $EXIT_CODE"
    echo "Check the error log for details: logs/download_docs_${SLURM_JOB_ID}.err"
fi

echo "=============================================="
echo "Job completed at $(date)"

exit $EXIT_CODE
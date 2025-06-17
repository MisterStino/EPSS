#!/bin/bash

# ============================================================================
# ML Pipeline Runner Script
# ============================================================================
# This script runs the complete machine learning pipeline from data preparation
# to model training and evaluation in the correct sequence.
#
# Pipeline Steps:
# 1. Data preprocessing and sorting (presort.py)
# 2. Arrow file conversion for efficient streaming (00_build_arrow.py) 
# 3. LSTM model training and evaluation (lstm_exp_window_eval.py)
#
# Usage: ./run_pipeline.sh
# Make executable: chmod +x run_pipeline.sh
# ============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to run a step with error handling
run_step() {
    local step_name="$1"
    local command="$2"
    local description="$3"
    
    log "Starting Step: ${BLUE}${step_name}${NC}"
    log "Description: ${description}"
    log "Command: ${command}"
    echo "----------------------------------------"
    
    if eval "$command"; then
        success "Step ${step_name} completed successfully"
        echo ""
    else
        error "Step ${step_name} failed with exit code $?"
        error "Pipeline execution stopped"
        exit 1
    fi
}

# Main pipeline execution
main() {
    log "${CYAN}🚀 Starting ML Pipeline Execution${NC}"
    log "Pipeline will run the following steps in sequence:"
    log "1. Data Preprocessing and Sorting"
    log "2. Arrow File Conversion" 
    log "3. LSTM Model Training and Evaluation"
    echo "========================================"
    echo ""
    
    # Check if we're in the right directory
    if [[ ! -f "ml_pipeline/data_prep/presort.py" ]]; then
        error "Please run this script from the project root directory"
        error "Expected to find: ml_pipeline/data_prep/presort.py"
        exit 1
    fi
    
    # Step 1: Data preprocessing and sorting
    run_step "1-PRESORT" \
        "python -m ml_pipeline.data_prep.presort" \
        "Preprocessing raw data, applying transformations, creating train/val/test splits, and sorting by CVE and date"
    
    # Step 2: Convert to Arrow format for efficient streaming
    run_step "2-ARROW" \
        "python -m ml_pipeline.data_prep.00_build_arrow" \
        "Converting sorted Parquet data to Arrow format for memory-efficient streaming during training"
    
    # Step 3: LSTM model training and evaluation
    run_step "3-TRAIN" \
        "python -m ml_pipeline.lstm_exp_window_eval" \
        "Training LSTM model with windowed evaluation and generating predictions"
    
    echo "========================================"
    success "${GREEN}🎉 Complete ML Pipeline Execution Finished Successfully!${NC}"
    log "All steps completed without errors"
    log "Check the results in:"
    log "  - ml_pipeline/work/ (intermediate files)"
    log "  - ml_pipeline/results/ (model outputs)"
    echo ""
}

# Run the pipeline
main "$@" 
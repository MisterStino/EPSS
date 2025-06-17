# ============================================================================
# ML Pipeline Runner Script (PowerShell Version)
# ============================================================================
# This script runs the complete machine learning pipeline from data preparation
# to model training and evaluation in the correct sequence.
#
# Pipeline Steps:
# 1. Data preprocessing and sorting (presort.py)
# 2. Arrow file conversion for efficient streaming (00_build_arrow.py) 
# 3. LSTM model training and evaluation (lstm_exp_window_eval.py)
#
# Usage: .\run_pipeline.ps1
# ============================================================================

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

# Function to run a step with error handling
function Invoke-PipelineStep {
    param(
        [string]$StepName,
        [string]$Command,
        [string]$Description
    )
    
    Write-Log "Starting Step: $StepName"
    Write-Log "Description: $Description"
    Write-Log "Command: $Command"
    Write-Host "----------------------------------------"
    
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command exited with code $LASTEXITCODE"
        }
        Write-Success "Step $StepName completed successfully"
        Write-Host ""
    }
    catch {
        Write-Error-Custom "Step $StepName failed: $($_.Exception.Message)"
        Write-Error-Custom "Pipeline execution stopped"
        exit 1
    }
}

# Main pipeline execution
function Start-Pipeline {
    Write-Log "🚀 Starting ML Pipeline Execution"
    Write-Log "Pipeline will run the following steps in sequence:"
    Write-Log "1. Data Preprocessing and Sorting"
    Write-Log "2. Arrow File Conversion"
    Write-Log "3. LSTM Model Training and Evaluation"
    Write-Host "========================================"
    Write-Host ""
    
    # Check if we're in the right directory
    if (-not (Test-Path "ml_pipeline\data_prep\presort.py")) {
        Write-Error-Custom "Please run this script from the project root directory"
        Write-Error-Custom "Expected to find: ml_pipeline\data_prep\presort.py"
        exit 1
    }
    
    # Step 1: Data preprocessing and sorting  
    Invoke-PipelineStep -StepName "1-PRESORT" `
        -Command "python -m ml_pipeline.data_prep.presort" `
        -Description "Preprocessing raw data, applying transformations, creating train/val/test splits, and sorting by CVE and date"
    
    # Step 2: Convert to Arrow format for efficient streaming
    Invoke-PipelineStep -StepName "2-ARROW" `
        -Command "python -m ml_pipeline.data_prep.00_build_arrow" `
        -Description "Converting sorted Parquet data to Arrow format for memory-efficient streaming during training"
    
    # Step 3: LSTM model training and evaluation
    Invoke-PipelineStep -StepName "3-TRAIN" `
        -Command "python -m ml_pipeline.lstm_exp_window_eval" `
        -Description "Training LSTM model with windowed evaluation and generating predictions"
    
    Write-Host "========================================"
    Write-Success "🎉 Complete ML Pipeline Execution Finished Successfully!"
    Write-Log "All steps completed without errors"
    Write-Log "Check the results in:"
    Write-Log "  - ml_pipeline\work\ (intermediate files)"
    Write-Log "  - ml_pipeline\results\ (model outputs)"
    Write-Host ""
}

# Run the pipeline
Start-Pipeline 
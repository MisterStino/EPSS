#!/usr/bin/env python3
"""
ML Pipeline Runner Module
=========================

This module runs the complete machine learning pipeline from data preparation
to model training and evaluation in the correct sequence.

Pipeline Steps:
1. Data preprocessing and sorting (presort.py)
2. Arrow file conversion for efficient streaming (00_build_arrow.py) 
3. Model training and evaluation (configurable: LSTM, TCN, or TCN-baseline)

Usage: 
  python -m ml_pipeline.run_pipeline                    # Default: LSTM
  python -m ml_pipeline.run_pipeline --model tcn        # TCN with SUS
  python -m ml_pipeline.run_pipeline --model tcn-baseline  # TCN without SUS
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


class PipelineRunner:
    """ML Pipeline Runner with error handling and logging"""
    
    def __init__(self, model_choice: str = "lstm"):
        self.steps_completed = 0
        self.total_steps = 7
        self.model_choice = model_choice
        
    def log(self, message: str) -> None:
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{Colors.CYAN}[{timestamp}]{Colors.NC} {message}")
    
    def error(self, message: str) -> None:
        """Log an error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}", file=sys.stderr)
    
    def success(self, message: str) -> None:
        """Log a success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
    
    def warning(self, message: str) -> None:
        """Log a warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
    
    def info(self, message: str) -> None:
        """Log an info message"""
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
    
    def run_step(self, step_name: str, module_name: str, description: str) -> bool:
        """
        Run a pipeline step with error handling
        
        Args:
            step_name: Human-readable step name
            module_name: Python module to execute
            description: Step description
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"Starting Step: {Colors.BLUE}{step_name}{Colors.NC}")
        self.log(f"Description: {description}")
        
        # Special handling for SUS weight quantile computation
        if module_name == "ml_pipeline.tools.compute_weight_quantile":
            cmd = [sys.executable, "-m", module_name, 
                   "--arrow", "ml_pipeline/work/epss_stage1.arrow",
                   "--beta", "3.0", "--look-ahead", "5", "--quantile", "0.995"]
            self.log(f"Command: python -m {module_name} --arrow ml_pipeline/work/epss_stage1.arrow --beta 3.0 --look-ahead 5 --quantile 0.995")
        else:
            cmd = [sys.executable, "-m", module_name]
            self.log(f"Command: python -m {module_name}")
        
        print("-" * 40)
        
        try:
            # Run the module as subprocess
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Let output go to console
                text=True
            )
            
            self.success(f"Step {step_name} completed successfully")
            self.steps_completed += 1
            print()
            return True
            
        except subprocess.CalledProcessError as e:
            self.error(f"Step {step_name} failed with exit code {e.returncode}")
            self.error("Pipeline execution stopped")
            return False
        except Exception as e:
            self.error(f"Step {step_name} failed with error: {str(e)}")
            self.error("Pipeline execution stopped")
            return False
    
    
    def run_pipeline(self) -> bool:
        """
        Run the complete ML pipeline
        
        Returns:
            True if all steps completed successfully, False otherwise
        """
        self.log(f"{Colors.CYAN}🚀 Starting ML Pipeline Execution{Colors.NC}")
        self.log("Pipeline will run the following steps in sequence:")
        self.log("-2. Data Sampling (Temporal Behavior)")
        self.log("-1. Final Sampling (Classification & Balancing)")
        self.log("0. Cleanup Stale Files")
        self.log("1. Data Preprocessing and Sorting")
        self.log("2. Arrow File Conversion")
        self.log("2B. SUS Weight Quantile Computation")
        self.log(f"3. {self.model_choice.upper()} Model Training and Evaluation")
        print("=" * 40)
        print()
        
        # Model-specific configuration
        model_configs = {
            "lstm": {
                "module": "ml_pipeline.lstm_exp_window_eval",
                "description": "Training LSTM model with SUS sampling and windowed evaluation"
            },
            "tcn": {
                "module": "ml_pipeline.tcn_exp_window_eval", 
                "description": "Training TCN model with SUS sampling and windowed evaluation"
            },
            "tcn-baseline": {
                "module": "ml_pipeline.tcn_exp_window_old_school",
                "description": "Training TCN baseline model (no SUS) for comparison"
            }
        }
        
        selected_model = model_configs.get(self.model_choice, model_configs["lstm"])
        
        # Define pipeline steps
        pipeline_steps = [
            {
                "name": "-2-SAMPLE",
                "module": "data.general_utils.sample",
                "description": "Creating temporal behavior-based sampling from full dataset using sample_by_temporal_behavior function"
            },
            {
                "name": "-1-FINAL-SAMPLE",
                "module": "data.general_utils.final_sample",
                "description": "Applying fixed classification logic and balanced sampling strategy with Type D correction"
            },
            {
                "name": "0-CLEANUP",
                "module": "ml_pipeline.cleanup_stale_files",
                "description": "Cleaning up stale files to ensure fresh pipeline run with latest data"
            },
            {
                "name": "1-PRESORT",
                "module": "ml_pipeline.data_prep.presort",
                "description": "Preprocessing raw data, applying transformations, creating train/val/test splits, and sorting by CVE and date"
            },
            {
                "name": "2-ARROW",
                "module": "ml_pipeline.data_prep.00_build_arrow",
                "description": "Converting sorted Parquet data to Arrow format for memory-efficient streaming during training"
            },
            {
                "name": "2B-SUS-Z",
                "module": "ml_pipeline.tools.compute_weight_quantile",
                "description": "Computing Z normalization constant for SUS sampling to ensure optimal spike detection"
            },
            {
                "name": "3-TRAIN",
                "module": selected_model["module"],
                "description": selected_model["description"]
            }
        ]
        
        # Run each step
        for step in pipeline_steps:
            # Skip SUS weight computation for baseline model (doesn't use SUS)
            if step["name"] == "2B-SUS-Z" and self.model_choice == "tcn-baseline":
                self.log(f"Skipping {step['name']} for baseline model (no SUS)")
                continue
            
            if not self.run_step(step["name"], step["module"], step["description"]):
                return False
        
        # All steps completed successfully
        print("=" * 40)
        self.success(f"{Colors.GREEN}🎉 Complete ML Pipeline Execution Finished Successfully!{Colors.NC}")
        self.log("All steps completed without errors")
        self.log("Check the results in:")
        self.log("  - ml_pipeline/work/ (intermediate files)")
        self.log("  - ml_pipeline/results/ (model outputs)")
        print()
        
        return True


def main():
    """Main entry point for the pipeline runner"""
    parser = argparse.ArgumentParser(description="Run the ML pipeline with different models")
    parser.add_argument("--model", choices=["lstm", "tcn", "tcn-baseline"], help="Specify the model to run", default="lstm")
    args = parser.parse_args()

    runner = PipelineRunner(args.model or "lstm")
    
    try:
        success = runner.run_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        runner.error("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
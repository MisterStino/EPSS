#!/usr/bin/env python3
"""
Cleanup Stale Files Script
==========================

This script removes all stale files that might interfere with fresh ML pipeline runs.
It ensures that only the latest, freshest data and outputs are used.

Usage: python -m ml_pipeline.cleanup_stale_files
"""

import os
import shutil
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def log(message: str) -> None:
    """Log a message with color"""
    print(f"{Colors.CYAN}[CLEANUP]{Colors.NC} {message}")


def warning(message: str) -> None:
    """Log a warning message"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def success(message: str) -> None:
    """Log a success message"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def error(message: str) -> None:
    """Log an error message"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def remove_file_if_exists(file_path: Path, description: str) -> bool:
    """Remove a file if it exists and log the action."""
    if file_path.exists():
        try:
            file_path.unlink()
            success(f"Removed {description}: {file_path}")
            return True
        except Exception as e:
            error(f"Failed to remove {description}: {file_path} - {e}")
            return False
    else:
        log(f"Not found (OK): {description} - {file_path}")
        return True


def remove_dir_if_exists(dir_path: Path, description: str) -> bool:
    """Remove a directory if it exists and log the action."""
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            success(f"Removed {description}: {dir_path}")
            return True
        except Exception as e:
            error(f"Failed to remove {description}: {dir_path} - {e}")
            return False
    else:
        log(f"Not found (OK): {description} - {dir_path}")
        return True


def cleanup_stale_files():
    """Main cleanup function to remove all stale files."""
    
    log("🧹 Starting comprehensive cleanup of stale files...")
    log("This ensures fresh pipeline runs with the latest data")
    print()
    
    # Files to remove from root directory (stale outputs)
    root_stale_files = [
        (Path("checkpoint.pt"), "Stale model checkpoint"),
        (Path("loss_history.csv"), "Stale training history"),
        (Path("predictions_stream.nc"), "Stale predictions file"),
    ]
    
    # Directories to remove from root (stale work directories)
    root_stale_dirs = [
        (Path("work"), "Stale work directory"),
    ]
    
    # ML Pipeline intermediate files (will be regenerated)
    ml_pipeline_files = [
        (Path("ml_pipeline/data_prep/work/epss_stage1.arrow"), "Intermediate Arrow file"),
        (Path("ml_pipeline/data_prep/work/vocab.json"), "Intermediate vocabulary"),
        (Path("ml_pipeline/data_prep/work/scaler.pkl"), "Intermediate scaler"),
        (Path("ml_pipeline/data_prep/work/splits.json"), "Intermediate splits info"),
        (Path("ml_pipeline/results/loss_history.csv"), "Previous training history"),
        (Path("ml_pipeline/results/checkpoint.pt"), "Previous model checkpoint"),
        (Path("ml_pipeline/results/predictions/predictions_stream.nc"), "Previous predictions"),
    ]
    
    # ML Pipeline intermediate directories
    ml_pipeline_dirs = [
        (Path("ml_pipeline/data_prep/work/epss_sorted"), "Intermediate sorted data"),
    ]
    
    # Python cache directories
    cache_dirs = [
        (Path("__pycache__"), "Root Python cache"),
        (Path("ml_pipeline/__pycache__"), "ML Pipeline cache"),
        (Path("ml_pipeline/data_prep/__pycache__"), "Data prep cache"),
        (Path("ml_pipeline/training/__pycache__"), "Training cache"),
    ]
    
    success_count = 0
    total_items = (len(root_stale_files) + len(root_stale_dirs) + 
                   len(ml_pipeline_files) + len(ml_pipeline_dirs) + len(cache_dirs))
    
    # Clean root directory stale files
    log("Cleaning stale files from root directory...")
    for file_path, description in root_stale_files:
        if remove_file_if_exists(file_path, description):
            success_count += 1
    
    # Clean root directory stale directories  
    log("Cleaning stale directories from root...")
    for dir_path, description in root_stale_dirs:
        if remove_dir_if_exists(dir_path, description):
            success_count += 1
    
    # Clean ML pipeline intermediate files
    log("Cleaning ML pipeline intermediate files...")
    for file_path, description in ml_pipeline_files:
        if remove_file_if_exists(file_path, description):
            success_count += 1
    
    # Clean ML pipeline intermediate directories
    log("Cleaning ML pipeline intermediate directories...")
    for dir_path, description in ml_pipeline_dirs:
        if remove_dir_if_exists(dir_path, description):
            success_count += 1
    
    # Clean Python cache directories
    log("Cleaning Python cache directories...")
    for dir_path, description in cache_dirs:
        if remove_dir_if_exists(dir_path, description):
            success_count += 1
    
    print()
    if success_count == total_items:
        success(f"🎉 Cleanup completed successfully! ({success_count}/{total_items} items)")
        success("✨ Your environment is now clean and ready for fresh pipeline runs")
    else:
        warning(f"⚠️  Cleanup completed with issues ({success_count}/{total_items} items)")
        warning("Some files may require manual removal or admin permissions")
    
    print()
    log("Next steps:")
    log("1. Run: python -m ml_pipeline.run_pipeline")
    log("2. All outputs will be in ml_pipeline/ directory")
    log("3. No more stale file conflicts!")


def main():
    """Main entry point for the cleanup script"""
    try:
        cleanup_stale_files()
    except KeyboardInterrupt:
        error("Cleanup interrupted by user")
        exit(1)
    except Exception as e:
        error(f"Unexpected error during cleanup: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main() 
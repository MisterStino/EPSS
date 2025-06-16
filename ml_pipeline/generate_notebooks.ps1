# Generate Notebooks Script
# Converts Python files to Jupyter notebooks using jupytext

Write-Host "GENERATING ML-PIPELINE NOTEBOOKS" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Check if jupytext is available
try {
    jupytext --version | Out-Null
    Write-Host "jupytext is available" -ForegroundColor Green
} catch {
    Write-Host "ERROR: jupytext not found!" -ForegroundColor Red
    Write-Host "Install with: pip install jupytext" -ForegroundColor Yellow
    exit 1
}

# Convert Python files to notebooks
Write-Host ""
Write-Host "Converting data_prep/00_build_arrow.py..." -ForegroundColor Yellow
jupytext --to notebook data_prep/00_build_arrow.py
if ($LASTEXITCODE -eq 0) { 
    Write-Host "SUCCESS: data_prep/00_build_arrow.ipynb created" -ForegroundColor Green 
} else { 
    Write-Host "FAILED: data_prep/00_build_arrow.py" -ForegroundColor Red 
}

Write-Host ""
Write-Host "Converting training/dataset_iterable.py..." -ForegroundColor Yellow  
jupytext --to notebook training/dataset_iterable.py
if ($LASTEXITCODE -eq 0) { 
    Write-Host "SUCCESS: training/dataset_iterable.ipynb created" -ForegroundColor Green 
} else { 
    Write-Host "FAILED: training/dataset_iterable.py" -ForegroundColor Red 
}

Write-Host ""
Write-Host "Converting lstm_exp_window_eval.py..." -ForegroundColor Yellow
jupytext --to notebook lstm_exp_window_eval.py  
if ($LASTEXITCODE -eq 0) { 
    Write-Host "SUCCESS: lstm_exp_window_eval.ipynb created" -ForegroundColor Green 
} else { 
    Write-Host "FAILED: lstm_exp_window_eval.py" -ForegroundColor Red 
}

Write-Host ""
Write-Host "NOTEBOOK GENERATION COMPLETE!" -ForegroundColor Green
Write-Host "All notebooks now contain the latest code from Python files." -ForegroundColor White 
# Cleanup script for fresh ml-pipeline deployment
Write-Host "🧹 Cleaning ml-pipeline for fresh deployment..."

# Remove generated files
$files = @(
    "predictions_stream.nc",
    "loss_history.csv", 
    "checkpoint.pt",
    "work/vocab.json",
    "work/scaler.pkl",
    "work/epss_stage1.arrow"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "✓ Removed: $file" -ForegroundColor Green
    } else {
        Write-Host "- Not found: $file" -ForegroundColor Yellow
    }
}

# Remove Python cache directories
$cacheDirs = @(
    "__pycache__",
    "training/__pycache__",
    "data_prep/__pycache__",
    "work/__pycache__"
)

foreach ($dir in $cacheDirs) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "✓ Removed: $dir" -ForegroundColor Green
    }
}

Write-Host "🎉 Cleanup complete! Ready for fresh deployment." -ForegroundColor Cyan 
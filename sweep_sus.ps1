# sweep_sus.ps1 - SUS hyperparameter grid search
# Run from repo root: .\sweep_sus.ps1

Write-Host "🔍 Starting SUS hyperparameter grid search..." -ForegroundColor Green
Write-Host "Testing 24 configurations: 4 β × 3 Δ × 2 Z-modes" -ForegroundColor Cyan

$betas  = 1,2,3,4
$deltas = 3,5,7
$zmode  = "quant", "max"       # Q0.995 vs running-max

$total = $betas.Length * $deltas.Length * $zmode.Length
$current = 0

# Ensure results directory exists
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

foreach ($b in $betas) {
  foreach ($d in $deltas) {
    foreach ($z in $zmode) {
      $current++
      Write-Host "[$current/$total] Testing β=$b, Δ=$d, Z=$z..." -ForegroundColor Yellow

      # 1️⃣ Build Z file
      if ($z -eq "quant") {
        Write-Host "  → Computing Z (quantile 99.5%)..."
        python -m ml_pipeline.tools.compute_weight_quantile `
               --arrow   ml_pipeline/work/epss_stage1.arrow `
               --beta    $b `
               --look-ahead $d `
               --quantile 0.995 `
               --output  ml_pipeline/work/sus_config_beta${b}_d${d}_q0.995.json | Out-Null
      } else {
        Write-Host "  → Computing Z (maximum)..."
        python -m ml_pipeline.tools.compute_weight_quantile `
               --arrow   ml_pipeline/work/epss_stage1.arrow `
               --beta    $b `
               --look-ahead $d `
               --quantile 1.0 `
               --output  ml_pipeline/work/sus_config_beta${b}_d${d}_q1.000.json | Out-Null
      }

      # 2️⃣ Two-epoch train
      $tag = "B${b}_D${d}_${z}"
      Write-Host "  → Training 2 epochs..."
      python -m ml_pipeline.lstm_exp_window_eval `
             --beta $b --look-ahead $d --epochs 2 `
             > results\grid_$tag.log 2>&1
      
      if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Finished $tag" -ForegroundColor Green
      } else {
        Write-Host "  ✗ Failed $tag" -ForegroundColor Red
      }
    }
  }
}

Write-Host "`n🎯 Grid search complete! Check results/ directory for logs and CSV files." -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Analyze CSV files to find best configuration" -ForegroundColor White
Write-Host "  2. Update SUS_CONFIG with winner parameters" -ForegroundColor White
Write-Host "  3. Run full training or Ray Tune" -ForegroundColor White 
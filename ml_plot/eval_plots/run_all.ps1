$files = @(
    "ml_plot\eval_plots\models\no_data_predictions_stream.nc",
    "ml_plot\eval_plots\models\lstm_v1_full_data.nc",
    "ml_plot\eval_plots\models\predictions_stream_sus_lstm.nc",
    "ml_plot\eval_plots\models\tcn_model_sus.nc",
    "ml_plot\eval_plots\models\stupid_predictions.nc"
)

foreach ($file in $files) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $csvOut = "ml_plot\eval_plots\outputs\$base\metrics_output.csv"
    $plotDir = "ml_plot\eval_plots\outputs\$base\plots"
    $predVsTrueDir = "$plotDir\pred_vs_true"

    # Create necessary directories
    New-Item -ItemType Directory -Path ("ml_plot\eval_plots\outputs\$base") -Force | Out-Null
    New-Item -ItemType Directory -Path $plotDir -Force | Out-Null
    New-Item -ItemType Directory -Path $predVsTrueDir -Force | Out-Null

    # Run the Python script
    python ml_plot\eval_plots\eval_plot.py $file `
    --true-var true `
    --pred-var pred `
    --out-csv $csvOut `
    --plot `
    --plot-dir $plotDir `
    --plot-preds-vs-true `
    --plot-preds-vs-true-dir $predVsTrueDir `
    --filter-recent-cves `
    --cves CVE-2013-1870 CVE-2024-8517 CVE-2024-3094
}

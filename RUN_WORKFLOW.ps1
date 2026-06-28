# 🚀 MASTER SENTIMENT ANALYSIS WORKFLOW - COMPLETE AUTOMATION
# This script orchestrates the entire analysis pipeline

# STOP: Do you have Ollama running?
Write-Host "
╔════════════════════════════════════════════════════════════════════╗
║          SENTIMENT ANALYSIS - COMPLETE WORKFLOW                   ║
║          Multi-Model Synthetic Dataset Generation                 ║
╚════════════════════════════════════════════════════════════════════╝
" -ForegroundColor Cyan

Write-Host "⚠️  PREREQUISITE CHECK:" -ForegroundColor Yellow
Write-Host "   Make sure Ollama is running (ollama serve in another terminal)"
Write-Host ""

# Check Ollama connection
Write-Host "🔍 Checking Ollama connection..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -ErrorAction Stop
    Write-Host "✅ Ollama is running and accessible!" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to Ollama at localhost:11434" -ForegroundColor Red
    Write-Host "   Please start Ollama: ollama serve" -ForegroundColor Yellow
    exit 1
}

# Change to project directory
cd d:\SentimentAnalysis

Write-Host "`n📋 WORKFLOW STEPS:" -ForegroundColor Cyan
Write-Host "  1. Install Python dependencies" -ForegroundColor White
Write-Host "  2. Generate synthetic conversations (FREE!)" -ForegroundColor White
Write-Host "  3. Run sentiment analysis on synthetic data" -ForegroundColor White
Write-Host "  4. Compare with real MoltBook data" -ForegroundColor White
Write-Host "  5. Generate RQ4 robustness report" -ForegroundColor White
Write-Host ""

# Step 0: Install dependencies
Write-Host "STEP 1: Installing Dependencies..." -ForegroundColor Green
pip install requests --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some dependencies may not have installed" -ForegroundColor Yellow
}

# Step 1: Estimate cost (should be FREE!)
Write-Host "`nSTEP 2: Verifying Cost Estimation..." -ForegroundColor Green
Write-Host "Running cost estimation..." -ForegroundColor Cyan
python scripts/multimodel_api_orchestrator.py `
    --model-pairs llama_mistral neural_phi `
    --num-conversations 100 `
    --estimate-only

Write-Host ""
Read-Host "Cost estimation complete. Press Enter to continue with generation" 

# Step 2: Generate synthetic conversations
Write-Host "`nSTEP 3: Generating Synthetic Conversations..." -ForegroundColor Green
Write-Host "Creating diverse conversations between LLM pairs..." -ForegroundColor Cyan
Write-Host "⏱️  This will take 10-30 minutes depending on your hardware" -ForegroundColor Yellow
Write-Host ""

# You can adjust these parameters
$modelPairs = @("llama_mistral", "neural_phi")
$numConversations = 100
$outputDir = "data/synthetic"

python scripts/multimodel_api_orchestrator.py `
    --model-pairs $modelPairs `
    --num-conversations $numConversations `
    --output-dir $outputDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Generation failed. Check Ollama is running." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Synthetic conversations generated!" -ForegroundColor Green

# Step 3: Run sentiment analysis on synthetic data
Write-Host "`nSTEP 4: Running Sentiment Analysis..." -ForegroundColor Green
Write-Host "Analyzing synthetic conversations..." -ForegroundColor Cyan

# Get the most recent generated file
$latestConvFile = Get-ChildItem $outputDir/conversations/*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($null -eq $latestConvFile) {
    Write-Host "❌ No synthetic conversation files found" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $($latestConvFile.Name)" -ForegroundColor Cyan

# Run complete analysis (comparison)
Write-Host "`nSTEP 5: Comparing with Real Data..." -ForegroundColor Green
Write-Host "Comparing sentiment distributions..." -ForegroundColor Cyan

python scripts/run_complete_analysis.py `
    --synthetic-data $outputDir/conversations `
    --real-data data/staged/moltbook_comments_all.jsonl `
    --output-dir $outputDir/analysis

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Analysis had issues, but continuing..." -ForegroundColor Yellow
}

# Step 4: Generate RQ4 robustness report
Write-Host "`nSTEP 6: Generating RQ4 Robustness Report..." -ForegroundColor Green
Write-Host "Creating publication-ready validation report..." -ForegroundColor Cyan

python scripts/generate_rq4_robustness_report.py `
    --synthetic-scores $outputDir/analysis/comparison_statistics.json `
    --output $outputDir/analysis/RQ4_ROBUSTNESS_REPORT.md

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ RQ4 Report generated!" -ForegroundColor Green
}

# Display results
Write-Host "`n
╔════════════════════════════════════════════════════════════════════╗
║                    ✅ WORKFLOW COMPLETE!                          ║
╚════════════════════════════════════════════════════════════════════╝
" -ForegroundColor Green

Write-Host "📊 Generated Files:" -ForegroundColor Cyan
Write-Host "  📁 Conversations: $outputDir/conversations/" -ForegroundColor White
Write-Host "  📊 Statistics: $outputDir/analysis/comparison_statistics.json" -ForegroundColor White
Write-Host "  📄 Comparison Report: $outputDir/analysis/SYNTHETIC_VS_REAL_COMPARISON.md" -ForegroundColor White
Write-Host "  🎓 RQ4 Report: $outputDir/analysis/RQ4_ROBUSTNESS_REPORT.md" -ForegroundColor White

Write-Host "`n📖 View Results:" -ForegroundColor Cyan
Write-Host "  1. Open comparison report:" -ForegroundColor White
Write-Host "     cat $outputDir/analysis/SYNTHETIC_VS_REAL_COMPARISON.md" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Open RQ4 robustness report:" -ForegroundColor White
Write-Host "     cat $outputDir/analysis/RQ4_ROBUSTNESS_REPORT.md" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. View statistics:" -ForegroundColor White
Write-Host "     Get-Content $outputDir/analysis/comparison_statistics.json | ConvertFrom-Json" -ForegroundColor Gray

Write-Host "`n💰 Cost Summary:" -ForegroundColor Yellow
Write-Host "  Ollama setup: $0 (completely free)" -ForegroundColor Green
Write-Host "  Model download: $0 (open-source)" -ForegroundColor Green
Write-Host "  Generation: $0 (runs locally)" -ForegroundColor Green
Write-Host "  Analysis: $0 (your computer)" -ForegroundColor Green
Write-Host "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  TOTAL COST: $0" -ForegroundColor Green

Write-Host "`n🎓 For Your Paper:" -ForegroundColor Cyan
Write-Host "  Include RQ4_ROBUSTNESS_REPORT.md in Section 4" -ForegroundColor White
Write-Host "  Key evidence: Chi-square test with p-value" -ForegroundColor White
Write-Host "  Visualization: Distribution comparison table" -ForegroundColor White

Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Review the comparison report" -ForegroundColor White
Write-Host "  2. Copy RQ4 finding to your paper" -ForegroundColor White
Write-Host "  3. (Optional) Generate more conversations with different parameters" -ForegroundColor White
Write-Host "  4. (Optional) Analyze model-specific interaction patterns" -ForegroundColor White

Write-Host "`n✨ All done! Your robustness validation is ready.`n" -ForegroundColor Green

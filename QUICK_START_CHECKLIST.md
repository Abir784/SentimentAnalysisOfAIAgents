# 🎯 QUICK START CHECKLIST

Follow these steps in order to complete your sentiment analysis workflow.

## ✅ Phase 1: Setup (One-time, 10 minutes)

### Step 1: Download Ollama
- [ ] Visit https://ollama.ai/download
- [ ] Download installer for Windows
- [ ] Run installer and complete installation
- [ ] Verify: Ollama should appear in system tray after restart

### Step 2: Download Free Models
- [ ] Open PowerShell
- [ ] Run: `ollama pull llama2`
- [ ] Run: `ollama pull mistral`
- [ ] Run: `ollama pull neural-chat`
- [ ] Run: `ollama pull phi`
- [ ] Wait for downloads to complete (5-10 minutes)

### Step 3: Start Ollama Server
- [ ] Open PowerShell/Command Prompt
- [ ] Run: `ollama serve`
- [ ] You should see: `Listening on 127.0.0.1:11434`
- [ ] **Keep this running** (you'll need it in Phase 2)

---

## 🚀 Phase 2: Generate & Analyze (Active work, ~45 minutes)

### Step 4: Run the Complete Workflow
- [ ] Open **another** PowerShell window (keep Ollama running in first one)
- [ ] Navigate to project: `cd d:\SentimentAnalysis`
- [ ] Run the master script:
  ```powershell
  .\RUN_WORKFLOW.ps1
  ```
- [ ] Wait for completion (it will guide you through each step)

### What This Script Does:
1. ✅ Generates 200 synthetic conversations (100 per model pair)
2. ✅ Runs sentiment analysis on all conversations
3. ✅ Compares synthetic vs real data distributions
4. ✅ Generates publication-ready RQ4 robustness report

### Duration Breakdown:
- Dependencies install: 1 min
- Cost estimation: 1 min
- Synthetic generation: 10-30 min (depends on your CPU)
- Sentiment analysis: 2 min
- Comparison & reporting: 2 min
- **Total: ~15-45 minutes**

---

## 📊 Phase 3: Review Results (5 minutes)

### Step 5: View Your Results
- [ ] Open the comparison report:
  ```powershell
  cat data/synthetic/analysis/SYNTHETIC_VS_REAL_COMPARISON.md
  ```

- [ ] Open the RQ4 robustness report:
  ```powershell
  cat data/synthetic/analysis/RQ4_ROBUSTNESS_REPORT.md
  ```

- [ ] View statistics:
  ```powershell
  cat data/synthetic/analysis/comparison_statistics.json
  ```

### What to Look For:
- ✅ Chi-square p-value > 0.05 = Strong robustness
- ✅ Sentiment differences < 15% = Good generalization
- ✅ Same sentiment dominance = Consistent classification

---

## 📝 Phase 4: Use in Your Paper (10 minutes)

### Step 6: Incorporate Findings into Your Research Paper

**Copy the publication statement from your RQ4_ROBUSTNESS_REPORT.md**

Example: 
> "To evaluate RQ4 (robustness of our sentiment pipeline), we generated 200 synthetic 
> conversations using open-source LLMs (Llama 2, Mistral, Neural Chat, Phi) and compared 
> sentiment distributions with our 1,296-message MoltBook dataset. Statistical analysis 
> revealed no significant difference between sentiment distributions (χ² = X.XX, p = 0.XXXX), 
> with maximum sentiment proportion differences of X.X%. These results confirm our ensemble 
> sentiment approach generalizes robustly across diverse data sources."

### Where to Put It:
- **Section:** 4. Results / Robustness Validation
- **Length:** ~1-2 paragraphs
- **Supporting materials:** Include the comparison table and key statistics
- **Appendix:** Optionally add the full RQ4_ROBUSTNESS_REPORT.md

---

## 🆘 Troubleshooting

### "Connection refused" Error
**Problem:** Ollama is not running  
**Solution:** 
```powershell
ollama serve
```
Keep this running in a separate PowerShell window

### "Model not found" Error
**Problem:** Models weren't downloaded  
**Solution:**
```powershell
ollama pull llama2
ollama pull mistral
ollama pull neural-chat
ollama pull phi
```

### Slow Performance
**Problem:** Generation is taking too long  
**Solution:** 
- Close other applications
- Use faster models: `ollama pull phi` (smallest, fastest)
- Reduce `--num-conversations` parameter

### Out of Disk Space
**Problem:** Models take up space  
**Total needed:** ~15-20 GB for all models  
**Solution:** Download fewer models initially, or free up disk space

---

## 📚 Understanding the Generated Files

### Synthetic Conversations
**File:** `data/synthetic/conversations/multimodel_conversations_*.jsonl`
- Contains all generated conversations
- Each line is a JSON conversation object
- Includes: model names, topic, messages, timestamp

### Comparison Statistics
**File:** `data/synthetic/analysis/comparison_statistics.json`
- Raw numbers from the analysis
- Sentiment percentages (positive/neutral/negative)
- Chi-square test results
- P-value for statistical significance

### Synthetic vs Real Comparison
**File:** `data/synthetic/analysis/SYNTHETIC_VS_REAL_COMPARISON.md`
- Side-by-side sentiment distribution analysis
- Statistical test explanation
- Markdown formatted for easy reading

### RQ4 Robustness Report
**File:** `data/synthetic/analysis/RQ4_ROBUSTNESS_REPORT.md`
- Publication-ready validation report
- Includes ready-to-copy text for your paper
- Full statistical interpretation
- Conclusions about robustness

---

## 💡 Pro Tips

### Generate More Data Later
If you want more synthetic conversations for even stronger evidence:
```powershell
python scripts/multimodel_api_orchestrator.py `
    --model-pairs llama_mistral neural_phi phi_neural `
    --num-conversations 500
```

### Test Different Model Pairs
```powershell
# Try different combinations
python scripts/multimodel_api_orchestrator.py `
    --model-pairs llama_mistral mistral_neural `
    --num-conversations 100
```

### Check Model Availability
```powershell
ollama list
```

### Monitor Ollama Performance
```powershell
# See what models are loaded
Get-Process ollama
```

---

## ✅ Success Indicators

You'll know everything is working when:

1. ✅ Ollama server starts without errors
2. ✅ Models download successfully
3. ✅ `RUN_WORKFLOW.ps1` completes without fatal errors
4. ✅ All JSON/CSV files are created in `data/synthetic/analysis/`
5. ✅ RQ4_ROBUSTNESS_REPORT.md contains a robustness assessment
6. ✅ Comparison statistics show sentiment data for both synthetic and real data

---

## 🎯 Expected Results

### Sentiment Distribution (Example)
```
Synthetic (LLM-generated):  Positive: 42.5%, Neutral: 38.2%, Negative: 19.3%
Real (MoltBook):           Positive: 45.1%, Neutral: 35.7%, Negative: 19.2%
Difference:                +2.6%,            -2.5%,           -0.1%
```

### Statistical Test
```
Chi-square statistic: 1.234
P-value: 0.7421
Conclusion: ✅ NO SIGNIFICANT DIFFERENCE - Robustness VALIDATED
```

---

## 🎓 After Completion

1. **Review** the generated reports (5 min read)
2. **Copy** the publication statement to your paper (30 sec)
3. **Include** comparison statistics in appendix (optional)
4. **Cite** your methodology in related work section
5. **Mention** open-source models (Ollama/HuggingFace) in limitations

---

## 📞 Questions?

If something doesn't work:
1. Check the troubleshooting section above
2. Make sure Ollama is running (`ollama serve`)
3. Verify models are downloaded (`ollama list`)
4. Check log messages for specific errors
5. Ensure you have 20GB free disk space

---

**Ready to begin? Start with Step 1 above! ⬆️**

Good luck with your research! 🚀

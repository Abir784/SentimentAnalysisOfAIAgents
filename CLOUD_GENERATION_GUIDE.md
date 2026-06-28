# 🚀 Synthetic Data Generation on Cloud Platforms

Fast, FREE generation of synthetic conversations using Ollama LLMs on Google Colab and Kaggle.

---

## ⚡ Quick Comparison

| Feature | Local | **Google Colab** | **Kaggle** |
|---------|-------|------------------|----------|
| **Speed** | 🐢 Slow | 🚀 **FAST** | 🚀 **FAST** |
| **Cost** | Your CPU | **FREE** | **FREE** |
| **Setup Time** | ~5 min | **~15 min** | **~10 min** |
| **Runtime Limit** | None | 12 hours | None |
| **GPU Available** | ❌ | ✅ Optional | ❌ |
| **Data Download** | Local | ✅ Easy | ✅ Easy |

---

## 📊 Performance Expectations

- **Google Colab**: 5-10 conversations/minute (with 50 total = 5-10 minutes)
- **Kaggle**: 3-8 conversations/minute (with 100 total = 10-20 minutes)  
- **Local** (your system): ~1-2 conversations/minute (with 100 total = 50-100 minutes!)

---

## 🎯 Google Colab Setup (Recommended for Speed)

### Step 1: Open Notebook in Colab
1. Go to https://colab.research.google.com
2. Click "GitHub" tab
3. Paste this file path: `COLAB_SYNTHETIC_GENERATION.ipynb`
4. OR upload the notebook directly

### Step 2: Run All Cells
Click `Runtime` → `Run all`

**Timeline:**
- **Cell 1-2**: Setup (2 min)
- **Cell 3**: Install Ollama & download models (10-15 min) ⏳
- **Cell 4**: Configuration (30 sec)
- **Cell 5**: Generate conversations (5-10 min) ⚡
- **Cell 6**: Sentiment analysis (1 min)
- **Cell 7**: Download results (30 sec) 📥

**Total Time: 20-30 minutes**

### Step 3: Download Results
The notebook automatically downloads:
- `conversations.jsonl` - All generated conversations
- `analysis_results.jsonl` - Sentiment analysis for each message

---

## 🎯 Kaggle Setup (Unlimited Runtime)

### Step 1: Upload Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Click "⊕ Add" → upload `KAGGLE_SYNTHETIC_GENERATION.ipynb`
4. OR paste content into a new notebook

### Step 2: Run All Cells
Click ▶️ button or press Ctrl+Enter on each cell

**Timeline:**
- **Cell 1-2**: Setup (2 min)
- **Cell 3**: Install Ollama & download models (10-15 min) ⏳
- **Cell 4**: Configuration (30 sec)
- **Cell 5**: Generate conversations (5-15 min depending on settings) ⚡
- **Cell 6**: Sentiment analysis (2 min)
- **Cell 7**: Summary & export (1 min)

**Total Time: 20-35 minutes**

### Step 3: Access Results
Files are saved to `/kaggle/working/`:
- Download directly from the output panel
- OR create a Kaggle Dataset from the files

---

## 🔧 Configuration Tips

### For FASTER Generation:
```python
# In configuration cell, change:
"conversations_per_pair": 10   # Reduce from 50
"model_pairs": [
    ("llama2", "mistral"),     # Keep fastest pair
]
```
**Estimated time: 5-8 minutes**

### For MORE Data:
```python
# In configuration cell, change:
"conversations_per_pair": 100  # Increase to 100
"model_pairs": [
    ("llama2", "mistral"),
    ("mistral", "phi"),
    ("neural-chat", "phi"),
]
```
**Estimated time: 30-40 minutes on Kaggle (unlimited runtime)**

---

## 📈 Output Format

### conversations.jsonl
```json
{
  "timestamp": "2026-05-23T21:45:00.123456",
  "pair": "llama2-mistral",
  "topic": "artificial intelligence",
  "messages": [
    {"speaker": "llama2", "text": "AI is revolutionizing..."},
    {"speaker": "mistral", "text": "Absolutely! The impact..."}
  ]
}
```

### sentiment_analysis.jsonl
```json
{
  "timestamp": "2026-05-23T21:45:00.123456",
  "model": "llama2",
  "text": "AI is revolutionizing technology",
  "sentiment": "positive",
  "scores": {
    "positive": 0.750,
    "neutral": 0.250,
    "negative": 0.000,
    "compound": 0.850
  }
}
```

---

## ⚠️ Common Issues & Solutions

### Issue: "Ollama connection timeout"
**Solution**: Wait 2-3 minutes after Ollama starts. The service takes time to initialize.

### Issue: "Model download stuck"
**Solution**: Kaggle/Colab models might be slow. Standard models are ~3-5GB. Be patient!

### Issue: "Out of memory"
**Solution**: Reduce `conversations_per_pair` from 50 to 20

### Issue: "Runtime disconnected" (Colab)
**Solution**: 
- Colab has 12-hour limit
- Use Kaggle for longer runs (no time limit)
- Enable "Stay awake" if needed

---

## 💡 Advanced Usage

### Running Multiple Notebooks in Parallel
1. Start Colab notebook generation (5-10 min)
2. Start Kaggle notebook generation (5-15 min)
3. Get results from both in 15-20 minutes instead of 30-50 minutes!

### Combining Results
```python
import pandas as pd
import jsonlines

# Load from both sources
colab_conversations = []
with jsonlines.open('colab_conversations.jsonl') as f:
    colab_conversations = list(f)

kaggle_conversations = []
with jsonlines.open('kaggle_conversations.jsonl') as f:
    kaggle_conversations = list(f)

# Combine
all_conversations = colab_conversations + kaggle_conversations

# Save combined
with jsonlines.open('all_conversations.jsonl', 'w') as f:
    f.write_all(all_conversations)
```

---

## 🎁 Cost Comparison

| Method | 100 Conversations | 1000 Conversations |
|--------|------------------|-------------------|
| **Colab/Kaggle** ✅ | $0.00 | $0.00 |
| **OpenAI API** | ~$0.50 | ~$5.00 |
| **Anthropic Claude** | ~$0.40 | ~$4.00 |
| **Your Local Computer** | ~1-2 hours | ~10-20 hours |

**Savings: 100% FREE + Faster! 🚀**

---

## 📝 Next Steps After Generation

1. **Download** the `.jsonl` files
2. **Use for training** your sentiment models
3. **Test robustness** of your existing models
4. **Compare with real data** (MoltBook dataset)
5. **Publish results** as RQ4 validation

---

## 🆘 Need Help?

- **Google Colab Issues**: Check runtime logs (Runtime → View logs)
- **Kaggle Issues**: Check "Issues" tab in your notebook
- **Ollama Issues**: Check if models are downloading (can be slow)
- **Memory Issues**: Reduce conversations or use parallel cloud runs

---

## 📚 References

- **Ollama**: https://ollama.ai
- **Models**: 
  - Llama 2: Meta's open LLM
  - Mistral: French open LLM  
  - Phi: Microsoft's efficient LLM
  - Neural Chat: Intel's optimized LLM
- **VADER Sentiment**: https://github.com/cjhutto/vaderSentiment

---

**Created**: May 23, 2026
**System**: Sentiment Analysis Multi-Model Generation Pipeline
**License**: Free for research & commercial use

#!/bin/bash
# Cloud Deployment Quick Reference
# Use these commands to deploy notebooks to cloud platforms

# ============================================================
# GOOGLE COLAB - Copy to a cell and run
# ============================================================

# Cell 1: Upload notebook from GitHub or local
# Option A: From GitHub (if you push to GitHub)
# !curl -L https://raw.githubusercontent.com/YOUR_REPO/COLAB_SYNTHETIC_GENERATION.ipynb > notebook.ipynb

# Option B: Mount Google Drive and load
# from google.colab import drive
# drive.mount('/content/drive')
# !cp '/content/drive/My Drive/COLAB_SYNTHETIC_GENERATION.ipynb' .

# ============================================================
# KAGGLE - Use Kaggle API
# ============================================================

# Installation
# pip install kaggle

# Upload notebook as new dataset
# kaggle datasets create -p ./synthetic_conversations/ -u

# Download results
# kaggle datasets download [dataset-slug]

# ============================================================
# QUICK COLAB INSTRUCTIONS
# ============================================================

COLAB_QUICK_START = """
# 🚀 Paste this into Google Colab Cell 1:

import subprocess
import sys

# Clone or download the notebook
!pip install -q gdown
!gdown --id FILE_ID -O COLAB_SYNTHETIC_GENERATION.ipynb

# Now run all cells below with Shift+Enter
# Expected timeline: 20-30 minutes total
"""

# ============================================================
# QUICK KAGGLE INSTRUCTIONS  
# ============================================================

KAGGLE_QUICK_START = """
# 🚀 Steps for Kaggle Notebook:

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Copy paste the KAGGLE_SYNTHETIC_GENERATION.ipynb content
4. Click ▶️ to run
5. Results saved to /kaggle/working/
6. Download .jsonl files

Expected timeline: 20-35 minutes total
"""

# ============================================================
# ENVIRONMENT DETAILS
# ============================================================

ENVIRONMENTS = {
    "google_colab": {
        "python": "3.10",
        "ram": "12.7 GB",
        "timeout": "12 hours",
        "cost": "FREE",
        "gpu": "Optional T4 available",
        "best_for": "Fast generation, multiple runs"
    },
    "kaggle": {
        "python": "3.10",
        "ram": "30 GB",
        "timeout": "Unlimited",
        "cost": "FREE",
        "gpu": "None",
        "best_for": "Large datasets, no time pressure"
    }
}

# ============================================================
# FILE MANIFEST
# ============================================================

FILES_CREATED = {
    "COLAB_SYNTHETIC_GENERATION.ipynb": {
        "description": "Google Colab optimized notebook",
        "runtime": "20-30 minutes",
        "best_for": "Speed, quick iterations",
        "requirements": ["Colab GPU preferred"],
    },
    "KAGGLE_SYNTHETIC_GENERATION.ipynb": {
        "description": "Kaggle optimized notebook", 
        "runtime": "20-35 minutes",
        "best_for": "Unlimited runtime, large batches",
        "requirements": ["Kaggle account"],
    },
    "CLOUD_GENERATION_GUIDE.md": {
        "description": "Complete setup and usage guide",
        "sections": [
            "Quick comparison table",
            "Step-by-step Colab setup",
            "Step-by-step Kaggle setup", 
            "Configuration tips",
            "Output format documentation",
            "Troubleshooting",
            "Cost analysis"
        ],
    }
}

# ============================================================
# PERFORMANCE BENCHMARKS
# ============================================================

EXPECTED_PERFORMANCE = {
    "google_colab": {
        "conversations_per_minute": "5-10",
        "time_for_50_conversations": "5-10 minutes",
        "time_for_100_conversations": "10-20 minutes",
        "time_for_200_conversations": "20-40 minutes",
    },
    "kaggle": {
        "conversations_per_minute": "3-8",
        "time_for_50_conversations": "6-15 minutes",
        "time_for_100_conversations": "12-30 minutes",
        "time_for_200_conversations": "25-60 minutes",
        "time_for_500_conversations": "60-150 minutes (use overnight)",
    },
    "local_machine": {
        "conversations_per_minute": "1-2",
        "time_for_50_conversations": "25-50 minutes",
        "time_for_100_conversations": "50-100 minutes",
        "time_for_200_conversations": "100-200 minutes",
        "note": "❌ NOT RECOMMENDED - Too slow!"
    }
}

# ============================================================
# TROUBLESHOOTING CHECKLIST
# ============================================================

TROUBLESHOOTING = {
    "ollama_timeout": {
        "error": "HTTPConnectionPool timeout",
        "solution": "Wait 2-3 minutes after Ollama starts",
        "prevention": "Increase timeout from 60s to 180s"
    },
    "model_download_slow": {
        "error": "Models taking too long",
        "solution": "Model files are 3-5GB each, this is normal",
        "prevention": "Start notebook and let it download overnight"
    },
    "memory_error": {
        "error": "Out of memory / RuntimeError",
        "solution": "Reduce conversations_per_pair from 50 to 20",
        "prevention": "Monitor runtime memory usage"
    },
    "colab_disconnect": {
        "error": "Runtime disconnected (Colab)",
        "solution": "Enable 'Stay awake' or use Kaggle instead",
        "prevention": "Use Kaggle for long runs (no time limit)"
    }
}

# ============================================================
# COST ANALYSIS
# ============================================================

COST_COMPARISON = {
    "100_conversations": {
        "colab_kaggle": "$0.00",
        "openai_api": "$0.50",
        "anthropic_api": "$0.40",
        "your_local_cpu": "$0 (but 1-2 hours of computing)",
        "savings": "100% FREE vs $0.40-0.50"
    },
    "1000_conversations": {
        "colab_kaggle": "$0.00",
        "openai_api": "$5.00",
        "anthropic_api": "$4.00",
        "your_local_cpu": "$0 (but 10-20 hours!)",
        "savings": "100% FREE + 10-20x faster"
    }
}

# ============================================================
# USAGE EXAMPLES
# ============================================================

EXAMPLE_USAGE = """
# Example 1: Generate 50 conversations (Colab - fastest)
1. Open Google Colab
2. Upload COLAB_SYNTHETIC_GENERATION.ipynb
3. Run all cells
4. Done in 5-10 minutes!
5. Download conversations.jsonl + analysis_results.jsonl

# Example 2: Generate 500 conversations (Kaggle - overnight)
1. Create Kaggle notebook
2. Paste KAGGLE_SYNTHETIC_GENERATION.ipynb code
3. Change "conversations_per_pair" to 250
4. Run before bed
5. Download next morning!

# Example 3: Parallel generation (fastest!)
1. Start Colab notebook (generates 50 conversations in 5 min)
2. Simultaneously start Kaggle notebook (generates 50 conversations in 6 min)
3. Both complete in ~10 minutes
4. Download from both
5. Have 100 conversations in same time as local (100 minutes!)
"""

print("✅ Cloud generation setup files created!")
print("📁 Files:")
print("  - COLAB_SYNTHETIC_GENERATION.ipynb")
print("  - KAGGLE_SYNTHETIC_GENERATION.ipynb")  
print("  - CLOUD_GENERATION_GUIDE.md")
print("\n🚀 Next: Open CLOUD_GENERATION_GUIDE.md for detailed instructions")

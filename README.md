# Campaign Segmentation & Statistical Analysis

A complete Python framework for identifying customer segments and testing statistical significance of campaign effects.

## 📋 What This Does

This analysis helps you answer questions like:
- **Which customer segments responded better to my campaign?**
- **Are the differences statistically significant or just random?**
- **Which segments should I target (or avoid) in future campaigns?**

## 🎯 Business Context

You ran a campaign on 80,000 customers with a 10% control group. Now you want to know:
1. Are there natural customer segments in your data?
2. Did the campaign work better/worse for different segments?
3. Which results are statistically reliable vs. just noise?

This framework uses:
- **K-means clustering** to find natural customer segments
- **Statistical hypothesis testing** to validate differences
- **Multiple visualization methods** to communicate findings

## 📊 Analysis Pipeline

```
Excel Data
    ↓
01_data_loading.py → Validate and load data
    ↓
02_preprocessing.py → Clean and encode features
    ↓
03_clustering.py → Find customer segments (run 3x, once per target)
    ↓
04_significance_testing.py → Test statistical significance (run 3x)
    ↓
Results: Significant segments identified!
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (you have 3.12.2 ✓)
- VS Code (installed ✓)
- Git Bash (installed ✓)

### Setup (5 minutes)

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your Excel file in `data/campaign_data.xlsx`
   - Verify column names match those in the scripts

3. **Run the analysis:**
   ```bash
   python main_analysis.py
   ```

**That's it!** Results will be in the `output/` folder.

👉 **For detailed setup instructions, see:** [QUICK_START.txt](QUICK_START.txt)

## 📁 Project Structure

```
campaign_analysis/
├── data/
│   └── campaign_data.xlsx          # Your input data
├── output/
│   ├── figures/                    # Visualizations (PNG files)
│   ├── results/                    # Detailed results (CSV files)
│   └── 00_analysis_summary.csv     # High-level summary - START HERE
├── 01_data_loading.py              # Load and validate data
├── 02_preprocessing.py             # Clean and prepare features
├── 03_clustering.py                # K-means clustering analysis
├── 04_significance_testing.py      # Statistical tests
├── main_analysis.py                # Run everything at once
├── requirements.txt                # Python packages needed
├── QUICK_START.txt                 # Detailed setup guide
└── RESULTS_GUIDE.txt               # How to interpret results
```

## 🔧 Configuration

### Your Data Requirements

Your Excel file needs these column types:

**1. Identifier (at least one):**
- `customer_id`, `account_number`, etc.

**2. Test/Control Flag:**
- `test_control` with values like 'test', 'control', or variants

**3. Feature Columns (for segmentation):**
- `collateral_status` - categorical (e.g., 'collateral', 'non-collateral')
- `nsol_flag` - boolean (True/False or 1/0)
- `balance_bin` - categorical (e.g., '0-1000', '1000-5000', '5000+')
- `last_rpc_months` - categorical (e.g., 'never', '0-1', '1-6', '6+')
- `last_payment_months` - categorical
- `time_since_chargeoff_months` - categorical

**4. Target Variables (outcomes you're measuring):**
- `recovery_percent` - continuous (0-100)
- `rpc_flag` - binary (1 = had RPC, 0 = no RPC)
- `payment_flag` - binary (1 = made payment, 0 = no payment)

### Adjusting Column Names

If your columns have different names, update these files:
- `01_data_loading.py` - search for "ADJUST THESE"
- `02_preprocessing.py` - search for "ADJUST THESE"

## 📈 Key Output Files

After running the analysis, check these files first:

### 1. Summary File
**`output/00_analysis_summary.csv`**
- Overall results for all target variables
- How many clusters found
- How many showed significance

### 2. Significant Clusters
**`output/04_significant_clusters_[target].csv`**
- The segments that showed statistically significant differences
- **These are your insights!**

### 3. Visualizations
**`output/figures/`**
- Elbow plots - choosing optimal number of clusters
- Silhouette plots - cluster quality
- Significance plots - which clusters had significant effects

### 4. Detailed Profiles
**`output/03_cluster_profiles_[target].csv`**
- Characteristics of each customer segment
- Average outcomes per cluster

## 🎓 Understanding the Analysis

### K-means Clustering

**What it does:** Groups similar customers together automatically

**Why it matters:** Instead of manually creating segments (e.g., "high balance + collateral"), the algorithm finds natural groupings in your data

**Example output:**
```
Cluster 0: 2,400 customers (30%) - High balance, collateral, recently contacted
Cluster 1: 1,800 customers (22%) - Low balance, non-collateral, never contacted
Cluster 2: 3,200 customers (40%) - Medium balance, mixed status
...
```

### Statistical Significance Testing

**What it does:** Tests if test vs. control differences are real or just luck

**Why it matters:** With small samples or small differences, you need statistics to know if you can trust the results

**Example output:**
```
Cluster 0: Test had 32.5% RPC vs Control 27.8% (p=0.018) ✓ Significant
Cluster 1: Test had 15.2% RPC vs Control 14.8% (p=0.672) ✗ Not significant
```

## 📊 Interpreting Results

### Reading Statistical Significance

- **p-value < 0.10:** 90% confident difference is real (*)
- **p-value < 0.05:** 95% confident difference is real (**)
- **p-value < 0.01:** 99% confident difference is real (***)
- **p-value ≥ 0.10:** Not confident - could be random

### Effect Size

- **Positive difference:** Test group performed better
- **Negative difference:** Control group performed better
- **Size matters:** +10% is more meaningful than +1%

### Sample Size

- Need adequate sample sizes for reliable tests
- Minimum: ~10 per group for proportions, ~30 for continuous
- Larger samples = more reliable results

👉 **For detailed interpretation help, see:** [RESULTS_GUIDE.txt](RESULTS_GUIDE.txt)

## 🔍 Common Use Cases

### Use Case 1: Which segments to target?
**Look for:** Clusters with positive, significant effects
**Action:** Prioritize these segments in next campaign

### Use Case 2: Which segments to exclude?
**Look for:** Clusters with negative, significant effects
**Action:** Exclude these segments to avoid wasting resources

### Use Case 3: Where to test more?
**Look for:** Large effects that aren't quite significant (small samples)
**Action:** Increase sample size in these segments next time

### Use Case 4: Campaign doesn't work?
**Look for:** No significant clusters
**Interpretation:** Campaign might not be effective, or effects are uniform
**Action:** Consider different messaging or targeting strategy

## 🛠️ Troubleshooting

### Script won't run
```bash
# Check Python is installed
python --version

# Reinstall packages
pip install -r requirements.txt

# Check you're in the right folder
pwd  # Should show campaign_analysis directory
```

### No significant results
- **This might be correct!** Most campaigns don't work universally
- Check control group size (need ~10% of total)
- Review if features are predictive

### Column errors
- Update column names in `01_data_loading.py` and `02_preprocessing.py`
- Verify your Excel file has all required columns

### Slow performance
- Reduce `max_k` in clustering (try 10 instead of 15)
- Close other programs
- Consider sampling data for initial exploration

## 📚 Additional Resources

- **[QUICK_START.txt](QUICK_START.txt)** - Detailed setup and running instructions
- **[RESULTS_GUIDE.txt](RESULTS_GUIDE.txt)** - How to interpret all outputs
- Code comments - Every script has extensive inline documentation

## 🤝 Getting Help

When asking for help, provide:
1. Which script you were running
2. Complete error message
3. What you expected vs. what happened

## 💡 Tips for Success

✓ Start with one target variable to learn the process
✓ Review outputs after each step
✓ Document your decisions (like choosing K)
✓ Don't worry if not all clusters are significant - that's normal
✓ Focus on actionable insights, not perfect models

## 📝 Analysis Checklist

- [ ] Data prepared in `data/campaign_data.xlsx`
- [ ] Column names updated in scripts (if needed)
- [ ] Packages installed: `pip install -r requirements.txt`
- [ ] Ran: `python main_analysis.py`
- [ ] Reviewed: `output/00_analysis_summary.csv`
- [ ] Reviewed: `output/04_significant_clusters_*.csv`
- [ ] Looked at visualizations in `output/figures/`
- [ ] Documented findings for stakeholders

## 🎯 Next Steps After Analysis

1. **Create executive summary** of findings
2. **Profile significant clusters** - who are they?
3. **Generate recommendations** - what actions to take?
4. **Plan next campaign** based on insights
5. **Share results** with stakeholders

---

## Quick Reference

**Run everything:**
```bash
python main_analysis.py
```

**Run step-by-step:**
```bash
python 01_data_loading.py
python 02_preprocessing.py
python 03_clustering.py        # Edit TARGET_VARIABLE first
python 04_significance_testing.py  # Edit TARGET_VARIABLE first
```

**Key results:**
- Summary: `output/00_analysis_summary.csv`
- Insights: `output/04_significant_clusters_*.csv`
- Visuals: `output/figures/*.png`

**Questions?** See [QUICK_START.txt](QUICK_START.txt) or [RESULTS_GUIDE.txt](RESULTS_GUIDE.txt)

---

Good luck with your analysis! 🚀

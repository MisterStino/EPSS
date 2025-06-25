### bare / not fine tuned classify model + raw mastodon discussions -> cvss severity
## 📝 **Vulnerability Classifier Evaluation Experiment Summary**

### **🔬 Experiment Design**
**Objective**: Evaluate VulnerabilitySeverityClassifier performance on social media text vs official NVD CVSS ratings

**Input Data:**
- **Source 1**: Mastodon CVE-centric posts (`mastodon_cve_centric_classifications.csv`)
  - 13,346 social media posts discussing specific CVEs
  - Text content from cybersecurity community discussions
- **Source 2**: NVD Ground Truth (`nvd_cve_ground_truth_20250608_114959.csv`)
  - 132,750 official CVSS scores/severity from 2021+ vulnerabilities
  - Authoritative vulnerability assessments

**Merged Dataset**: 13,059 CVE records with both social media text + official CVSS labels

### **🤖 Model & Methodology**
**Classifier**: `CIRCL/vulnerability-severity-classification-roberta-base` (RoBERTa-based)
- **Device**: GPU (RTX 3050 Ti) 
- **Batch Size**: 50 records per batch
- **Input**: Raw Mastodon post text content
- **Outputs**: 
  - Severity labels: low/medium/high/critical
  - CVSS scores: 0-10 numerical values

**Evaluation Setup**:
- Inner join on CVE IDs between social media + ground truth
- Case-sensitive label matching (fixed post-analysis)
- Standard classification & regression metrics

### **📊 Results Summary**
**Classification (Severity Prediction)**:
- **Overall Accuracy**: 52.3% (vs 25% random baseline)
- **Per-Class Accuracy**: HIGH (75.3%) > MEDIUM (54.0%) > CRITICAL (7.7%) > LOW (0.3%)
- **Key Bias**: Over-predicts HIGH (62% vs 40% actual), under-predicts CRITICAL/LOW

**Regression (CVSS Score Prediction)**:
- **MAE**: 1.35 points (moderate error)
- **Within ±1.0**: 49.2% of predictions
- **R²**: 0.083 (weak correlation)

### **🔑 Key Findings**
1. **Social media partially predicts official severity** (52% vs random 25%)
2. **Strong HIGH-severity detection** (75% accuracy)
3. **Poor extreme severity detection** (CRITICAL: 7.7%, LOW: 0.3%)
4. **Systematic bias toward moderate-high severity predictions**

### **⚠️ Technical Notes**
- **Critical Bug**: Initial 0% accuracy due to case sensitivity (fixed)
- **Data Quality**: 100% successful predictions, no processing failures
- **Ground Truth Filter**: Excluded 2 "NONE" severity records for fair comparison

### **📁 Output Files**
- `vulnerability_classifier_evaluation_20250608_121319.csv` (predictions + ground truth)
- `evaluation_metrics_20250608_121319.txt` (detailed performance metrics)

**For Future Comparisons**: Different social media sources, newer CVE data, alternative vulnerability classifiers, or modified severity mappings.



###

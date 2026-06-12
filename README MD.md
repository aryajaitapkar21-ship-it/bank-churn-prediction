# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/shubhendra7/european-bank-customer-churn)

> A complete machine learning pipeline that predicts bank customer churn, assigns risk probability scores, and delivers explainable business insights — deployed as an interactive Streamlit dashboard.

---

## 📌 Project Overview

Customer churn directly impacts:
- Customer Lifetime Value (CLV)
- Revenue stability
- Cross-sell and upsell potential
- Long-term competitiveness of retail banks

This project introduces a **predictive churn intelligence system** that assigns risk probabilities to customers before they leave, enabling proactive and targeted retention strategies.

**Dataset:** European Bank — 10,000 customers from France, Spain & Germany  
**Source:** [Kaggle — European Bank Customer Churn](https://www.kaggle.com/datasets/shubhendra7/european-bank-customer-churn)

---

## 📊 Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 80.6% | 0.480 | 0.768 |
| Decision Tree | 78.4% | 0.520 | 0.734 |
| Random Forest | 86.2% | 0.620 | 0.856 |
| **Gradient Boosting ★** | **86.9%** | **0.650** | **0.873** |

**Overall churn rate:** 20.37% (2,037 / 10,000 customers)

---

## 🗂️ Repository Structure

```
bank-churn-prediction/
│
├── data/
│   └── European_Bank.csv          # Raw dataset (10,000 customers)
│
├── src/
│   └── app.py                     # Streamlit dashboard (5 modules)
│
├── notebooks/
│   └── churn_analysis.ipynb       # Full EDA + modelling notebook
│
├── models/
│   └── model_training.py          # Standalone model training script
│
├── results/
│   ├── eda_summary.md             # EDA findings summary
│   └── model_comparison.md        # Model performance comparison
│
├── docs/
│   ├── Research_Paper_IEEE.docx   # IEEE format research paper
│   └── Executive_Summary.docx    # Government stakeholder summary
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-churn-prediction.git
cd bank-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard
```bash
streamlit run src/app.py
```

### 4. Run model training standalone
```bash
python models/model_training.py
```

---

## 📱 Streamlit Dashboard Features

| Tab | Description |
|-----|-------------|
| 📊 **EDA** | Full exploratory analysis — 6 interactive charts |
| 🤖 **Model Comparison** | ROC curves, confusion matrices, metrics table |
| 🔍 **Feature Importance** | Top-20 features, SHAP-style bar charts, correlation heatmap |
| 🎯 **What-If Simulator** | Live sliders → real-time churn probability + risk factors |
| 📋 **Customer Risk Table** | All 10,000 customers scored, filtered, downloadable CSV |

---

## 🔑 Key Findings

### High-Risk Segments
- 🔴 **Germany**: 32.4% churn — 2× France/Spain baseline
- 🔴 **Age 51-60**: 56.2% churn — most critical demographic cohort  
- 🔴 **Inactive Members**: 26.9% vs 14.3% for active members
- 🔴 **Female Customers**: 25.1% vs 16.5% for male customers (52% gap)
- 🔴 **3-4 Products**: 82-100% churn — evidence of forced cross-selling

### Protective Factors
- 🟢 **Ages 18-30**: Only 7.5% churn — loyal young base
- 🟢 **Active Members**: Engagement = strongest controllable retention lever
- 🟢 **2 Products**: Optimal engagement, lowest churn rate
- 🟢 **France & Spain**: Stable ~16% baseline churn

---

## 📈 Top Churn Predictors (Feature Importance)

1. **Age** — Life-stage transition risk (especially 41-60)
2. **Account Balance** — Financial stress indicator
3. **IsActiveMember** — Engagement as protective factor
4. **Balance-to-Salary Ratio** *(engineered)* — Relative financial dependency
5. **Geography** — Germany structural risk
6. **NumOfProducts** — Over-selling signal at 3+
7. **CreditScore** — Financial vulnerability proxy

---

## 💰 Business Impact

```
Current Annual Churn Loss:  €5.09M  (2,037 × €2,500 avg CLV)
25% Reduction Saves:        €1.27M+  (509 customers retained)
Retention Program ROI:      4–6×     (vs campaign cost)
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn** — ML models
- **Streamlit** — Interactive dashboard
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualisation
- **Jupyter** — Exploratory notebook

---

## 📄 Deliverables

| Deliverable | Description |
|-------------|-------------|
| `src/app.py` | Production-ready Streamlit dashboard |
| `notebooks/churn_analysis.ipynb` | Full EDA + ML notebook |
| `docs/Research_Paper_IEEE.docx` | IEEE-format research paper |
| `docs/Executive_Summary.docx` | Stakeholder executive summary |
| `models/model_training.py` | Reusable training pipeline |

---

## 📚 References

- Friedman, J.H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Scikit-learn: Pedregosa et al., JMLR 12, pp. 2825–2830, 2011.
- European Central Bank Statistical Data Warehouse, 2025.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Bank Churn Intelligence Project**  
European Bank Dataset · Kaggle · 2025

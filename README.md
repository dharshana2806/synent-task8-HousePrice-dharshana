# 🏠 House Price Prediction — Advanced ML Project
### Task 8 | Synent Data Science Internship

---

## 📌 Problem Statement
Predict the **sale price of a house** using its physical attributes, location, quality, and engineered features. The goal is to train and compare multiple regression models and identify the best performer.

---

## 📊 Dataset
- **File:** `house_prices.csv`
- **Rows:** 1,460 houses
- **Columns:** 21 raw features → 54 after engineering
- **Target:** `SalePrice` (in USD)

### Key Raw Features
| Feature | Description |
|---------|-------------|
| OverallQual | Overall quality rating (1–10) |
| GrLivArea | Ground living area (sq ft) |
| TotalBsmtSF | Total basement area |
| GarageCars | Garage car capacity |
| YearBuilt | Year of construction |
| Neighborhood | Location (one-hot encoded) |
| SalePrice | **TARGET** — sale price in USD |

---

## ⚙️ Feature Engineering (10 New Features)
| New Feature | Formula | Why It Matters |
|------------|---------|----------------|
| `TotalSF` | Bsmt + 1st + 2nd floor | Total usable space |
| `TotalBath` | FullBath + 0.5×HalfBath | Realistic bathroom count |
| `HouseAge` | YrSold − YearBuilt | Older = cheaper |
| `Remodeled` | YearRemodAdd ≠ YearBuilt | Renovation adds value |
| `HasGarage` | GarageArea > 0 | Binary garage presence |
| `TotalPorch` | Deck + Open + Enclosed | Total outdoor space |
| `Qual_Area` | OverallQual × GrLivArea | **Most powerful feature** |
| `Qual_TotalSF` | OverallQual × TotalSF | Quality-adjusted total SF |
| `Neighborhood_*` | One-hot encoding | Location premium/discount |
| `SalePrice_Log` | log1p(SalePrice) | Removes skewness in target |

---

## 🤖 Models Trained

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Lasso Regression** ⭐ | **0.9529** | **$16,163** | **$13,023** |
| Linear Regression | 0.9528 | $16,169 | $13,052 |
| Ridge Regression | 0.9517 | $16,358 | $13,144 |
| Gradient Boosting | 0.9460 | $17,299 | $13,792 |
| Random Forest | 0.9220 | $20,793 | $16,617 |
| Decision Tree | 0.8656 | $27,293 | $21,189 |

> **R² = 0.95** means the model explains **95% of price variation**

---

## 🌟 Top Features by Importance
1. `Qual_Area` — Quality × Area interaction ⭐
2. `OverallQual` — Overall house quality
3. `TotalSF` — Total square footage
4. `HouseAge` — Age at time of sale
5. `Neighborhood_*` — Location

---

## 📁 Project Files
```
house_price_project/
│
├── house_prices.csv              ← Raw dataset
├── model_results.csv             ← Model comparison table
├── House_Price_Prediction.ipynb  ← Full notebook (run this)
├── eda_dashboard.png             ← EDA visualisations
├── model_comparison.png          ← Model performance charts
├── feature_importance.png        ← Feature importance bar chart
└── README.md                     ← This file
```

---

## 🚀 How to Run
1. Clone / download and extract the project folder
2. Open `House_Price_Prediction.ipynb` in Jupyter or Google Colab
3. Run all cells top to bottom
4. All plots and results will generate automatically

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🛠️ Tools Used
| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models & evaluation |
| matplotlib | Visualisations |
| seaborn | Statistical plots |
| Jupyter Notebook | Interactive environment |

---

## 💡 Key Insights
- **Log-transforming** SalePrice reduces skewness and improves all model performances
- **Interaction feature** `Qual_Area` (quality × area) is the single strongest predictor
- **Location** (Neighborhood) accounts for $15K–$50K premium/discount
- **House Age** has a clear negative correlation with price
- Linear models (Lasso/Ridge) outperform tree models here due to strong linear relationships after log transform

---


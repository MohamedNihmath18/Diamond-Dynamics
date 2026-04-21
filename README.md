# 💎 Diamond Dynamics: Price Prediction & Market Segmentation

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-green?style=for-the-badge)

---

## 📌 Project Overview

The diamond market is driven by multiple quality attributes such as **carat, cut, clarity, color**, and physical dimensions. This project builds an end-to-end Machine Learning solution that:

- 💰 **Predicts diamond prices** in INR and USD using regression models and ANN
- 🔮 **Segments the diamond market** into meaningful groups using K-Means Clustering
- 📱 **Deploys an interactive Streamlit web app** for real-time predictions

This project was developed as part of the **GUVI Data Science Certification Program**.

---

## 🎯 Objectives

- Predict diamond prices using multiple ML regression models and an Artificial Neural Network (ANN)
- Cluster diamonds into market segments based on physical and qualitative features
- Deploy an interactive Streamlit UI for price prediction and cluster identification

---

## 🌍 Real-World Use Cases

| Use Case | Description |
|---|---|
| 💹 Dynamic Pricing | Helps diamond retailers set competitive prices |
| 📦 Inventory Management | Categorizes diamonds into market segments for product listing |
| 🎁 Recommendation Engine | Suggests diamonds based on customer budget and preference |
| 🎯 Customer Segmentation | Enables personalized marketing strategies |

---

## 📁 Project Structure

```
Diamond-Dynamics/
│
├── 📂 data/
│   ├── diamonds.csv                  # Raw dataset (53,940 rows)
│   ├── diamonds_clean.csv            # After preprocessing & outlier handling
│   ├── diamonds_engineered.csv       # After feature engineering & encoding
│   └── diamonds_final.csv            # Final dataset with cluster labels
│
├── 📂 notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data cleaning, outlier handling
│   ├── 02_eda.ipynb                  # Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb  # Feature creation & selection
│   ├── 04_regression_models.ipynb    # ML regression models
│   ├── 05_ann_model.ipynb            # ANN (MLPRegressor) model
│   └── 06_clustering.ipynb           # K-Means clustering & segmentation
│
├── 📂 models/
│   ├── best_regression_model.pkl     # Random Forest (best model)
│   ├── ann_model.pkl                 # ANN MLPRegressor model
│   ├── kmeans_model.pkl              # K-Means clustering model
│   ├── cluster_names.pkl             # Cluster ID to name mapping
│   ├── scaler.pkl                    # Scaler for regression features
│   └── scaler_cluster.pkl            # Scaler for clustering features
│
├── 📂 app/
│   └── app.py                        # Streamlit web application
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Seaborn built-in diamonds dataset |
| Rows | 53,940 |
| Columns | 10 |
| Target | `price` (converted to INR) |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `carat` | Float | Weight of the diamond — primary price driver |
| `cut` | Ordinal | Cut quality: Fair → Good → Very Good → Premium → Ideal |
| `color` | Ordinal | Color grade: D (best) → J (worst) |
| `clarity` | Ordinal | Clarity grade: IF (best) → I1 (worst) |
| `depth` | Float | Total depth percentage = z / mean(x,y) × 100 |
| `table` | Float | Width of top facet as % of average diameter |
| `price` | Int | Price in USD — **Target Variable** |
| `x` | Float | Length in mm |
| `y` | Float | Width in mm |
| `z` | Float | Depth in mm |

---

## 🔧 Project Pipeline

```
Raw Data
   │
   ▼
01 Data Preprocessing
   ├── Handle missing values
   ├── Fix zero values in x, y, z
   ├── Remove 146 duplicates
   └── Cap outliers using IQR (5,471 outliers handled)
   │
   ▼
02 Exploratory Data Analysis
   ├── Distribution plots
   ├── Boxplots & correlation heatmap
   ├── Pairplot & scatterplots
   └── Skewness analysis
   │
   ▼
03 Feature Engineering & Selection
   ├── USD → INR conversion (rate: 84.0)
   ├── New features: Volume, Price/Carat, Dimension Ratio, Carat Category
   ├── Log transformation on skewed features
   ├── Ordinal encoding for cut, color, clarity
   └── Correlation matrix for feature selection
   │
   ▼
04 Regression Models          05 ANN Model
   ├── Linear Regression          └── MLPRegressor
   ├── Decision Tree                  128 → 64 → 32 → 1
   ├── Random Forest ✅ Best
   ├── KNN
   └── XGBoost
   │
   ▼
06 Clustering
   ├── Elbow Method
   ├── Silhouette Score
   ├── KMeans (k=3)
   └── PCA Visualization
   │
   ▼
Streamlit App
   ├── Price Prediction Module
   ├── Market Segment Module
   └── Visual Insights Dashboard
```

---

## 📈 Model Performance

### Regression Models

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| 🥇 Random Forest | 0.0036 | 0.0133 | **0.9998** |
| 🥈 XGBoost | 0.0096 | 0.0157 | 0.9998 |
| 🥉 Decision Tree | 0.0058 | 0.0188 | 0.9996 |
| KNN | 0.0656 | 0.0929 | 0.9914 |
| Linear Regression | 0.0702 | 0.1109 | 0.9877 |
| ANN (MLPRegressor) | 0.0430 | 0.3557 | 0.8738 |

> ✅ **Best Model: Random Forest** with R² = 0.9998 and RMSE = 0.0133

### Clustering Results

| Metric | Value |
|---|---|
| Algorithm | K-Means |
| Optimal K | 3 |
| Inertia Score | 341,305.48 |
| Silhouette Score | 0.2298 |

---

## 💎 Market Segments

| Cluster | Segment Name | Avg Carat | Avg Price (USD) | Avg Price (INR) | Count |
|---|---|---|---|---|---|
| 0 | 💎 Premium Heavy Diamonds | 1.63 | $10,714 | ₹8,99,965 | 8,672 |
| 2 | 💍 Mid-range Balanced Diamonds | 0.92 | $4,483 | ₹3,76,575 | 20,433 |
| 1 | 💰 Affordable Small Diamonds | 0.40 | $1,096 | ₹92,076 | 24,689 |

---

## ⚙️ Feature Engineering

| New Feature | Formula | Purpose |
|---|---|---|
| `price_inr` | price × 84.0 | Convert USD to INR |
| `volume` | x × y × z | Physical size of diamond |
| `price_per_carat` | price / carat | Value density |
| `dimension_ratio` | (x+y) / (2×z) | Shape proportions |
| `carat_category` | Light/Medium/Heavy | Categorical size grouping |
| `log_*` columns | log1p(col) | Reduce skewness |

---

## 📉 Skewness Treatment

| Feature | Before | After | Method |
|---|---|---|---|
| price | 1.6182 🔴 | 0.1149 ✅ | Log Transform |
| carat | 1.1137 🔴 | 0.5787 ✅ | Log Transform |
| y | 2.4687 🔴 | 0.2018 ✅ | Log Transform |
| z | 1.5896 🔴 | 0.1983 ✅ | Log Transform |

---

## 📱 Streamlit App Features

### Tab 1 — 💰 Price Prediction
- Input diamond attributes (carat, cut, color, clarity, dimensions)
- Predicts price in both **INR** and **USD**
- Uses best performing **Random Forest** model

### Tab 2 — 🔮 Market Segment
- Identifies which market category the diamond belongs to
- Displays cluster name with description and emoji
- Uses trained **K-Means** model

### Tab 3 — 📊 Visual Insights
- Market segment distribution bar chart
- Average price per segment
- Average carat per segment

---

## 🚀 How to Run

### 1. Clone or download the project
```bash
cd "Diamond-Dynamics"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```
01_data_preprocessing.ipynb
02_eda.ipynb
03_feature_engineering.ipynb
04_regression_models.ipynb
05_ann_model.ipynb
06_clustering.ipynb
```

### 4. Launch Streamlit app
```bash
cd app
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| ANN | MLPRegressor (Scikit-learn) |
| Clustering | KMeans, PCA |
| Web App | Streamlit |
| Model Saving | Joblib |

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
joblib
scipy
```

---

## 🔍 Key Insights from EDA

- 💎 **Carat** is the strongest price predictor with correlation **0.9365**
- 📐 **x, y, z dimensions** are highly correlated with price (>0.95)
- ✂️ **Ideal cut** is most common but doesn't have the highest average price
- 🎨 **J color** (worst grade) has higher avg price — driven by larger carat sizes
- 🔬 **I1 clarity** (worst grade) has higher avg price — driven by carat size
- 📊 **Price distribution** is highly right-skewed → log transformation applied

---

## 👨‍💻 Author

**Mohamed Nihmath**
Data Analyst → Aspiring Data Scientist
GUVI Data Science Certification Program

---

## 📄 License

This project is developed for educational purposes as part of the GUVI HCL Data Science program.

---

*💎 Diamond Dynamics | GUVI Data Science Project | Built with Python & Streamlit*

# 🚖 NYC Taxi Demand Forecasting (Region-wise | 15-Min Intervals)

This repository contains the **production-ready forecasting pipeline** for predicting NYC taxi pickup demand at **15-minute granularity**, across **30 dynamically clustered regions**.  
It is the final, optimized version of the experimentation work completed in the R&D repository.

**Live Project:**  
https://apoorvtechh-dashboard-demand-prediction-app-gx2szx.streamlit.app/ 
## 🌟 Project Summary

The system forecasts **future taxi demand** in each NYC region by combining:

- **Region-wise time-series modeling**
- **Long-term trend extraction**
- **Short-term spike learning**
- **Hybrid forecasting (Prophet + XGBoost)**

This reduces the error from **0.047 → 0.0301 MAPE**, enabling **more accurate** demand predictions.

The project is designed for **scalability, automation, and ML observability** using:

- **DVC** — data versioning & reproducible pipelines  
- **MLflow** — experiment tracking & model registry  
- **Per-region model training** — 30 individual models, each optimized  
- **Fully modular pipeline**— ingestion → clustering → features → training → evaluation  

---

## 📌 Live Synopsis (Streamlit App)

🔗 **https://apoorvtechh-synopsis-dp-app-wxbcw6.streamlit.app/**

This interactive app showcases:

- Dataset exploration  
- Region clustering visualizations  
- Feature engineering  
- Prophet baseline  
- XGBoost modeling  
- Hybrid model improvements  
- Region-wise performance comparison  

---

## 📚 Experimentation Repository

All initial R&D, notebooks, and experimental modeling are here:

🔗 **https://github.com/apoorvtechh/demand_preidction_experimentation**

This includes:

- 33M-row preprocessing  
- KMeans region discovery  
- EWMA optimization  
- Rolling/lag feature testing  
- Baseline model comparisons  
- Hybrid model experiments  
- Region-wise forecasting behavior  

---

## 🏗️ Production Pipeline Overview

The production system in this repository contains a complete ML workflow:

### **1️⃣ Data Ingestion**
- Reads raw NYC taxi files
- Removes coordinate & fare outliers
- Saves cleaned Parquet for downstream stages

### **2️⃣ Feature Extraction**
- StandardScaler + MiniBatchKMeans clustering  
- Assigns region IDs to all 33M pickups  
- Converts data into 15-minute aggregated time-series  
- Computes optimal EWMA smoothing per region  

### **3️⃣ Feature Processing**
- Lag features (`t-1` to `t-4`)  
- Rolling statistics (mean/std windows: 3 & 6)  
- Date-based features (day, month)  
- Train/test creation (Jan–Feb → Train, March → Test)

### **4️⃣ Model Training**
- Per-region Prophet trend extraction (hourly)  
- Trend merged into 15-min data  
- XGBoost model trained with trend + engineered features  
- All region models saved in `/models/training`

### **5️⃣ Model Evaluation**
- Region-wise MAPE calculation  
- Generates final evaluation report in `/models/evaluation`

### **6️⃣ Model Registration (MLflow)**
- Prophet & XGBoost models logged and versioned  
- Artifacts stored in remote MLflow backend  

---

## 🔁 DVC Pipeline (End-to-End Automation)

Every stage of the system runs through DVC:


This ensures:
- Reproducible machine learning  
- Version-controlled data and models  
- Easy dependency tracking  
- Fully automated retraining  

---

## 📦 Folder Structure (Simplified)
demand_forecasting/

│

├── data/

│   ├── raw/

│   ├── interim/

│   └── processed/

│

├── models/

│   ├── training/

│   └── evaluation/

│

├── src/

│   ├── data/

│   ├── features/

│   ├── models/

│   └── utils/

│

├── dvc.yaml

├── params.yaml

└── README.md

---

## 📈 Final Model Performance

| Model | Avg MAPE |
|-------|----------|
| Baseline XGBoost | **0.0370** |
| Hybrid Prophet + XGBoost | **0.0301** |
| Best Regions | **0.018–0.022 MAPE** |

The hybrid system consistently outperforms classical ML and time-series models alone.

---

## 🎯 Purpose & Real-world Use

This forecasting system helps:

- **Drivers** → move toward hotspots before demand peaks  
- **Taxi platforms** → reduce wait times & improve fleet distribution  
- **Cities** → understand mobility patterns & improve transport planning  

The architecture supports **scalable production deployment**, automated retraining, and versioned ML operations.

---

## 👨‍💻 Author

**Apoorv Gupta**  
LinkedIn: https://www.linkedin.com/in/apoorv-gupta-32117b226  


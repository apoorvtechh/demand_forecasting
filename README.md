# NYC Taxi Demand Forecasting (Region-wise | 15-Min Intervals)

This repository contains the **production-ready forecasting pipeline** for predicting NYC taxi pickup demand at **15-minute granularity**, across **30 dynamically clustered regions**.  
It represents the final, fully optimized implementation derived from extensive large-scale experimentation and R&D.

**Live Project Dashboard:**  
https://apoorvtechh-dashboard-demand-prediction-app-gx2szx.streamlit.app/

---

## Project Summary

The system forecasts **future NYC taxi pickup demand** independently for each region by combining:

- Region-wise time-series modeling  
- Long-term trend and seasonality extraction  
- Short-term spike, volatility, and anomaly learning  
- A hybrid forecasting architecture using **Prophet + XGBoost**

By separating trend learning and residual learning, the system reduces forecasting error from **0.047 → 0.0301 MAPE**, resulting in significantly more accurate, stable, and production-reliable demand predictions.

The project is architected for **scalability, automation, reproducibility, and ML observability**, leveraging the following:

- **DVC** for dataset versioning, pipeline orchestration, and full reproducibility  
- **MLflow** for experiment tracking, metric logging, artifact storage, and model registry  
- **Per-region model training**, with **30 independently trained and optimized models**  
- **Fully modular pipeline**, spanning ingestion → spatial clustering → feature engineering → training → evaluation → deployment  

---

## Live Synopsis (Streamlit App)

https://apoorvtechh-synopsis-dp-app-wxbcw6.streamlit.app/

The interactive synopsis application provides a complete walkthrough of the project, including:

- Raw dataset exploration and preprocessing overview  
- NYC geospatial demand distribution and region clustering visualizations  
- Feature engineering strategy and temporal signal extraction  
- Prophet-based baseline time-series modeling  
- XGBoost-based machine learning modeling  
- Hybrid model integration and step-by-step error reduction analysis  
- Region-wise performance breakdown and comparative evaluation  

---

## Forecasting Architecture

The forecasting system follows a hybrid, region-wise architecture designed to capture both global and local demand dynamics.

- **Spatial segmentation:**  
  NYC is divided into **30 demand regions** using **MiniBatch KMeans clustering** on pickup latitude and longitude, ensuring scalable clustering on 33M+ records.

- **Temporal resolution:**  
  Demand is aggregated at **15-minute intervals**, enabling fine-grained, near real-time demand forecasting.

- **Trend and seasonality modeling:**  
  **Prophet** is used to model long-term trends, daily and weekly seasonality, and recurring temporal patterns within each region.

- **Residual and spike learning:**  
  **XGBoost** learns short-term deviations, sudden demand spikes, volatility patterns, and residual errors not captured by Prophet.

- **Final prediction formulation:**  
  Final demand prediction is computed as:  
  `Final Demand = Prophet Trend + XGBoost Residual Correction`

This separation of responsibilities results in improved robustness during both stable periods and high-volatility demand windows.

---

## Experimentation Repository

All early-stage research, large-scale preprocessing, and model experimentation were conducted separately to maintain a clean production codebase.

https://github.com/apoorvtechh/demand_preidction_experimentation

This experimentation repository includes:

- Large-scale preprocessing and aggregation of **33M+ NYC Yellow Taxi trip records**  
- Exploratory data analysis of temporal, spatial, and seasonal demand patterns  
- Demand region clustering strategy experiments  
- Baseline statistical and machine learning forecasting models  
- Feature engineering iterations, lag analysis, rolling statistics, and ablation studies  
- Error analysis and model comparison across regions  

---

## System Highlights

- End-to-end production-grade forecasting pipeline  
- Region-wise independent model optimization for improved accuracy  
- Hybrid time-series and machine learning modeling strategy  
- Reproducible data and model pipelines powered by DVC  
- Full experiment tracking, artifact management, and model registry using MLflow  
- Real-time, interactive visualization through Streamlit dashboards  
- Scalable architecture suitable for urban mobility and ride-hailing platforms  

---

## License

This project is intended for educational, research, and portfolio demonstration purposes.

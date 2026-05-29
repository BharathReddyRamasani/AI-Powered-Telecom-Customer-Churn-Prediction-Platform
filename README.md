# 📡 AI-Powered Telecom Customer Churn Prediction Platform

An enterprise-grade customer intelligence and retention dashboard. Powered by a **Random Forest Classifier** to predict individual churn probabilities, a **Business Logic Expected Revenue Loss Model**, and a **K-Means Clustering + Principal Component Analysis (PCA)** pipeline to segment accounts into dynamic personas.

This platform empowers marketing, finance, and customer success teams to proactively address customer attrition, prioritize outreach by high-value accounts at risk, and deliver tailored retention offers.

---

## 🚀 Key Features

1. **📊 Tab 1: Executive Dashboard (BI)**
   - **Financial Vulnerability Tracking**: Key KPIs showcasing overall churn rates, monthly contract values, and **Expected Monthly Churn Loss** ($P(\text{Churn}) \times \text{Monthly Charges}$).
   - **Interactive Visualizations**: Distro charts of customer cohorts and churn across contract durations.
   - **ML Explainability**: Relative feature importances identifying what factors (e.g. Month-to-month contracts, Fiber optic connection, billing method) drive churn.

2. **🔮 Tab 2: Customer Risk Simulator**
   - **Real-Time Inference**: Dynamic dropdowns and sliders representing a customer profile.
   - **Risk Gauge**: Instant status display (Critical, High, Moderate, Low Risk) based on custom thresholding.
   - **Cohort Persona Identifier**: Identifies the K-Means cluster the simulated customer matches.
   - **Automated Retention Recommendations**: Real-time action cards providing concrete outreach strategies (e.g. contract upgrades, auto-pay incentives, connection diagnostics, custom trials).

3. **👥 Tab 3: Cohorts & Customer Segmentation**
   - **2D PCA Projection Space**: Visualizes the high-dimensional customer base in 2D. 
   - **Interactive Simulated Point Mapping**: Plots the active simulated customer as a star icon in real-time, showing exactly where they fit in the customer landscape.
   - **Persona Profiles**: Detailed statistical summary of the 4 cluster segments.

4. **📥 Tab 4: Batch Outreach Target Generator**
   - **Revenue-at-Risk Sorting**: Sorts customer cohorts by expected revenue loss to maximize marketing ROI.
   - **Probability Cutoffs**: Slider to narrow down outreach target lists.
   - **Outreach Exporter**: Export a CSV list containing contact targets, risk scores, monthly charges, and recommendation actions.

---

## 📂 Project Architecture

```
├── data/
│   └── Telco-Customer-Churn.csv         # Raw customer dataset (Kaggle)
├── src/
│   └── train.py                         # Cleaning, training, and clustering script
├── models/
│   ├── churn_model.pkl                  # Serialized Scikit-Learn Random Forest Pipeline
│   ├── kmeans_model.pkl                 # K-Means clustering model
│   ├── pca_model.pkl                    # Fitted PCA transformation (2 components)
│   ├── preprocessor_for_clustering.pkl  # ColumnTransformer preprocessor for clustering
│   ├── col_specs.json                   # List of numeric and categorical column lists
│   ├── metrics.json                     # Serialized classification evaluation scores
│   └── cluster_profiles.json            # Statistics and descriptions of K-Means cohorts
├── app.py                               # Core Streamlit Web Application
├── requirements.txt                     # System Python package dependencies
└── README.md                            # Documentation and deployment guide
```

---

## 📈 Model Performance & Customer Cohorts

### 1. Classification Performance (Test Set)
- **Accuracy**: `77.40%`
- **Precision**: `56.57%`
- **Recall (Sensitivity)**: `64.44%` *(Prioritized in churn forecasting to capture maximum vulnerable accounts)*
- **F1 Score**: `60.25%`
- **ROC-AUC**: `82.51%`

### 2. K-Means Customer Cohorts
The dataset was partitioned into 4 distinct groups:
* **Cluster 0: Premium Loyalists**
  * *Characteristics*: High monthly charges (avg: \$91.18), long-term contracts (1/2 years), high tenure (avg: 59.5 months), low churn rate (13.4%).
  * *Focus*: Cross-sell premium add-ons and value-added services.
* **Cluster 1: Standard Savers**
  * *Characteristics*: Low monthly charges (avg: \$21.08), long-term contracts, moderate tenure (avg: 30.7 months), very low churn rate (7.4%).
  * *Focus*: Maintain relationships and offer low-tier upgrades.
* **Cluster 2: High-Risk Tech-Savvy**
  * *Characteristics*: High monthly charges (avg: \$84.59), month-to-month contracts, low tenure (avg: 15.6 months), very high churn (57.0%).
  * *Focus*: Critical retention target. Proactive connection health audits and contract upgrade discount offers.
* **Cluster 3: Budget Month-to-Month**
  * *Characteristics*: Moderate monthly charges (avg: \$50.66), month-to-month contracts, moderate tenure (avg: 20.6 months), moderate churn (25.5%).
  * *Focus*: Shift to auto-pay and offer budget bundle packages.

---

## 🛠️ Local Setup Guide

### 1. Installation
Clone the repository, open a terminal in the folder, and run:
```bash
pip install -r requirements.txt
```

### 2. Run Model Training
Execute the training script to ingest the dataset, train models, and serialize artifacts:
```bash
python src/train.py
```
This generates the `models/` directory and all model binaries.

### 3. Launch Streamlit Application
Start the Streamlit local development web server:
```bash
streamlit run app.py
```
The app will open automatically in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment to Streamlit Cloud

1. Push this codebase to your public GitHub repository:
   `https://github.com/BharathReddyRamasani/AI-Powered-Telecom-Customer-Churn-Prediction-Platform.git`
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**, select this repository, select the `main` branch, and set the entry file to `app.py`.
4. Click **Deploy**. Streamlit Cloud will parse `requirements.txt`, install dependencies, run the app, and expose a public URL!
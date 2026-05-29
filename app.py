import os
import json
import joblib
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. PAGE CONFIG & THEME SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Analytics Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium glassmorphic dark-mode custom styles
st.markdown("""
<style>
    /* Dark Theme Background and Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Low-profile Horizontal Header styling (Space Efficient) */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 60%, #3b82f6 100%);
        color: white;
        padding: 0.85rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.15);
        border: 1px solid #1e3a8a;
    }
    .app-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.2rem;
        font-weight: 400;
        line-height: 1.3;
    }
    
    /* Sleek Slate Cards (Space Efficient) */
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        border-color: #475569;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.725rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Alerts for Churn Risk Simulator */
    .risk-banner {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    .risk-banner-critical {
        background-color: #2d1616;
        border-left-color: #ef4444;
        color: #fca5a5;
        border: 1px solid #ef444440;
    }
    .risk-banner-high {
        background-color: #2c1a10;
        border-left-color: #f97316;
        color: #fed7aa;
        border: 1px solid #f9731640;
    }
    .risk-banner-medium {
        background-color: #2b2512;
        border-left-color: #eab308;
        color: #fef08a;
        border: 1px solid #eab30840;
    }
    .risk-banner-low {
        background-color: #12251a;
        border-left-color: #22c55e;
        color: #bbf7d0;
        border: 1px solid #22c55e40;
    }
    
    .recommendation-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rec-title {
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.15rem;
        color: #60a5fa;
    }
    .rec-desc {
        font-size: 0.775rem;
        color: #cbd5e1;
        line-height: 1.35;
    }
    
    /* Segment Persona Badge */
    .persona-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 1.15rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #334155;
    }
    
    /* Tabs font color adaptation */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: 700;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. UTILITY TO STYLE MATPLOTLIB PLOTS IN DARK THEME
# ---------------------------------------------------------
def apply_dark_plot_style(ax, fig):
    # Set backgrounds
    fig.patch.set_facecolor('#1e293b')
    ax.set_facecolor('#1e293b')
    
    # Grid and borders
    ax.spines['bottom'].set_color('#475569')
    ax.spines['top'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['right'].set_color('#475569')
    ax.grid(True, color='#334155', linestyle='--', alpha=0.4)
    
    # Label colors
    ax.xaxis.label.set_color('#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.title.set_color('#f1f5f9')
    ax.tick_params(colors='#94a3b8', which='both', labelsize=8.5)
    
    # Legend color
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor('#1e293b')
        legend.get_frame().set_edgecolor('#475569')
        for text in legend.get_texts():
            text.set_color('#cbd5e1')
            text.set_size(8)

# ---------------------------------------------------------
# 3. LOAD DATA AND MODELS (WITH CACHING & RELATIVE PATHS)
# ---------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv('data/Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df['SeniorCitizen_Str'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

@st.cache_resource
def load_ml_models():
    models_dir = 'models'
    churn_model = joblib.load(os.path.join(models_dir, 'churn_model.pkl'))
    kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
    pca_model = joblib.load(os.path.join(models_dir, 'pca_model.pkl'))
    preprocessor_clustering = joblib.load(os.path.join(models_dir, 'preprocessor_for_clustering.pkl'))
    
    with open(os.path.join(models_dir, 'cluster_profiles.json'), 'r') as f:
        cluster_profiles = json.load(f)
        
    with open(os.path.join(models_dir, 'col_specs.json'), 'r') as f:
        col_specs = json.load(f)
        
    with open(os.path.join(models_dir, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    return churn_model, kmeans_model, pca_model, preprocessor_clustering, cluster_profiles, col_specs, metrics

# Load components
try:
    df_raw = load_raw_data()
    churn_model, kmeans_model, pca_model, preprocessor_clustering, cluster_profiles, col_specs, metrics = load_ml_models()
    models_loaded = True
except Exception as e:
    st.error(f"Failed to load models or dataset. Please run python src/train.py first. Error: {e}")
    models_loaded = False

# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/telecommunication-tower.png", width=60)
st.sidebar.markdown("### **Telecom Churn Engine**")
st.sidebar.write("An enterprise customer intelligence platform powered by Random Forest Classification and K-Means Clustering.")

if models_loaded:
    st.sidebar.divider()
    st.sidebar.markdown("#### **System Health**")
    st.sidebar.success(f"Models Active (F1: {metrics['f1_score']:.2f})")
    st.sidebar.info(f"Active Customers Analyzed: {len(df_raw):,}")

# ---------------------------------------------------------
# 5. MAIN INTERFACE HEADER (SPACE EFFICIENT HORIZONTAL)
# ---------------------------------------------------------
st.markdown(textwrap.dedent("""
<div class="app-header">
    <div style="display: flex; align-items: center; gap: 0.85rem;">
        <span style="font-size: 1.8rem; line-height: 1;">📡</span>
        <div>
            <h1 class="app-title">AI-Powered Telecom Customer Churn Prediction Platform</h1>
            <p class="app-subtitle">Identify customer churn risks, calculate monthly revenue vulnerability, segment accounts, and deploy retention strategies in real time.</p>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)

if models_loaded:
    # Setup tabs
    tab_dashboard, tab_simulator, tab_cohorts, tab_batch = st.tabs([
        "📊 Executive Dashboard", 
        "🔮 Customer Risk Simulator", 
        "👥 Cohorts & Segments", 
        "📥 Batch Analysis & Export"
    ])
    
    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE DASHBOARD
    # ---------------------------------------------------------
    with tab_dashboard:
        st.markdown('<div class="section-title">Operational & Revenue Metrics</div>', unsafe_allow_html=True)
        
        # Calculate global parameters
        total_customers = len(df_raw)
        actual_churn_rate = (df_raw['Churn'] == 'Yes').mean()
        total_monthly_revenue = df_raw['MonthlyCharges'].sum()
        
        # Predict on all dataset using classification model to find Expected Revenue Loss
        X_all = df_raw.drop(columns=['customerID', 'Churn'])
        X_all['SeniorCitizen'] = X_all['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Run classification pipeline
        churn_probs = churn_model.predict_proba(X_all)[:, 1]
        expected_monthly_loss = np.sum(churn_probs * df_raw['MonthlyCharges'])
        
        # Render KPI cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #60a5fa;">{total_customers:,}</div>
                <div class="metric-label">Total Accounts</div>
            </div>
            """), unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #f87171;">{actual_churn_rate:.1%}</div>
                <div class="metric-label">Historical Churn</div>
            </div>
            """), unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #34d399;">${total_monthly_revenue/1e3:.1f}k</div>
                <div class="metric-label">Monthly Contract Value</div>
            </div>
            """), unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(textwrap.dedent(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #fb923c;">${expected_monthly_loss/1e3:.1f}k</div>
                <div class="metric-label">Monthly Expected Churn Loss</div>
            </div>
            """), unsafe_allow_html=True)
            
        st.markdown('<div class="section-title">Churn Vulnerability & Drivers</div>', unsafe_allow_html=True)
        
        # Column split for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Churn distribution pie chart
            fig, ax = plt.subplots(figsize=(6, 3.8))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            
            churn_counts = df_raw['Churn'].value_counts()
            wedges, texts, autotexts = ax.pie(
                churn_counts, 
                labels=['Stayed', 'Churned'], 
                autopct='%1.1f%%',
                startangle=90, 
                colors=['#3b82f6', '#ef4444'],
                wedgeprops=dict(width=0.45, edgecolor='#334155'),
                textprops=dict(color='#cbd5e1', fontweight='bold', fontsize=9.5)
            )
            for autotext in autotexts:
                autotext.set_color('white')
            
            ax.set_title("Customer Cohort Status (Actual)", fontsize=11, fontweight='bold', pad=10, color='#f1f5f9')
            st.pyplot(fig)
            
            # Contract type churn analysis
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            contract_churn = pd.crosstab(df_raw['Contract'], df_raw['Churn'], normalize='index') * 100
            contract_churn.plot(kind='bar', stacked=True, color=['#3b82f6', '#ef4444'], ax=ax2, edgecolor='#334155')
            
            ax2.set_title("Churn Rate by Contract Type", fontsize=11, fontweight='bold', pad=10, color='#f1f5f9')
            ax2.set_ylabel("Percentage (%)", fontsize=8.5)
            ax2.set_xlabel("Contract Type", fontsize=8.5)
            ax2.legend(["Stayed", "Churned"], loc="lower left", fontsize=8)
            plt.xticks(rotation=0)
            apply_dark_plot_style(ax2, fig2)
            st.pyplot(fig2)
            
        with chart_col2:
            # Model Feature Importance
            fig3, ax3 = plt.subplots(figsize=(6, 8.5))
            
            classifier = churn_model.named_steps['classifier']
            preprocessor = churn_model.named_steps['preprocessor']
            
            cat_encoder = preprocessor.named_transformers_['cat']
            encoded_cat_cols = list(cat_encoder.get_feature_names_out(col_specs['categorical_cols']))
            all_feature_names = col_specs['numeric_cols'] + encoded_cat_cols
            
            importances = classifier.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True).tail(12)
            
            feat_imp_df['Feature'] = feat_imp_df['Feature'].str.replace('cat__', '').str.replace('num__', '')
            
            colors = ['#60a5fa' if x < feat_imp_df['Importance'].max() * 0.7 else '#2563eb' for x in feat_imp_df['Importance']]
            ax3.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color=colors, edgecolor='#334155')
            ax3.set_title("Key Drivers of Customer Churn (RF Importance)", fontsize=11, fontweight='bold', pad=10, color='#f1f5f9')
            ax3.set_xlabel("Relative Predictive Power", fontsize=8.5)
            apply_dark_plot_style(ax3, fig3)
            st.pyplot(fig3)
            
    # ---------------------------------------------------------
    # TAB 2: CUSTOMER RISK SIMULATOR
    # ---------------------------------------------------------
    with tab_simulator:
        st.markdown('<div class="section-title">Real-Time Risk & Recommendation Engine</div>', unsafe_allow_html=True)
        
        sim_col_inputs, sim_col_results = st.columns([1, 1.2])
        
        with sim_col_inputs:
            st.markdown("### 🔧 Customer Attributes")
            
            exp_demo = st.expander("👤 1. Demographics", expanded=True)
            with exp_demo:
                sim_gender = st.selectbox("Gender", ["Female", "Male"])
                sim_senior = st.selectbox("Senior Citizen (Age >= 65)", ["No", "Yes"])
                sim_partner = st.selectbox("Has Partner", ["Yes", "No"])
                sim_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
                
            exp_services = st.expander("🔌 2. Subscribed Services", expanded=True)
            with exp_services:
                sim_phone = st.selectbox("Phone Service", ["Yes", "No"])
                if sim_phone == "Yes":
                    sim_multiline = st.selectbox("Multiple Phone Lines", ["No", "Yes"])
                else:
                    sim_multiline = "No phone service"
                    
                sim_internet = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
                
                if sim_internet != "No":
                    sim_security = st.selectbox("Online Security Add-on", ["No", "Yes"])
                    sim_backup = st.selectbox("Online Backup Add-on", ["No", "Yes"])
                    sim_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
                    sim_techsupport = st.selectbox("Premium Tech Support Add-on", ["No", "Yes"])
                    sim_tv = st.selectbox("Streaming TV Service", ["No", "Yes"])
                    sim_movies = st.selectbox("Streaming Movies Service", ["No", "Yes"])
                else:
                    sim_security = "No internet service"
                    sim_backup = "No internet service"
                    sim_protection = "No internet service"
                    sim_techsupport = "No internet service"
                    sim_tv = "No internet service"
                    sim_movies = "No internet service"
                    
            exp_billing = st.expander("💳 3. Billing & Contract", expanded=True)
            with exp_billing:
                sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                sim_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                sim_payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                sim_tenure = st.slider("Account Tenure (Months)", 1, 72, 12)
                sim_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 75.0)
                
                calculated_total = float(sim_monthly * sim_tenure)
                sim_total = st.number_input("Total Charges ($)", min_value=0.0, value=calculated_total, step=50.0)
                
            input_dict = {
                "gender": sim_gender,
                "SeniorCitizen": sim_senior,
                "Partner": sim_partner,
                "Dependents": sim_dependents,
                "tenure": sim_tenure,
                "PhoneService": sim_phone,
                "MultipleLines": sim_multiline,
                "InternetService": sim_internet,
                "OnlineSecurity": sim_security,
                "OnlineBackup": sim_backup,
                "DeviceProtection": sim_protection,
                "TechSupport": sim_techsupport,
                "StreamingTV": sim_tv,
                "StreamingMovies": sim_movies,
                "Contract": sim_contract,
                "PaperlessBilling": sim_paperless,
                "PaymentMethod": sim_payment,
                "MonthlyCharges": sim_monthly,
                "TotalCharges": sim_total
            }
            sim_df = pd.DataFrame([input_dict])
            
        with sim_col_results:
            st.markdown("### 📊 Predicted Account Status")
            
            prob_churn = churn_model.predict_proba(sim_df)[0][1]
            
            if prob_churn >= 0.65:
                risk_cat = "Critical Risk"
                risk_style = "risk-banner-critical"
                risk_color = "#ef4444"
                risk_icon = "🚨"
            elif prob_churn >= 0.40:
                risk_cat = "High Risk"
                risk_style = "risk-banner-high"
                risk_color = "#f97316"
                risk_icon = "⚠️"
            elif prob_churn >= 0.20:
                risk_cat = "Moderate Risk"
                risk_style = "risk-banner-medium"
                risk_color = "#eab308"
                risk_icon = "⚡"
            else:
                risk_cat = "Low Risk / Stable"
                risk_style = "risk-banner-low"
                risk_color = "#22c55e"
                risk_icon = "✅"
                
            exp_loss_val = prob_churn * sim_monthly
            
            clustering_input_df = sim_df.copy()
            encoded_sim = preprocessor_clustering.transform(clustering_input_df)
            
            sim_cluster = int(kmeans_model.predict(encoded_sim)[0])
            sim_pca_coords = pca_model.transform(encoded_sim)[0]
            
            sim_profile = next(item for item in cluster_profiles if item["cluster_id"] == sim_cluster)
            
            badge_colors = ["#3b82f6", "#10b981", "#ef4444", "#fb923c"]
            badge_bg = ["#1e3a8a30", "#064e3b30", "#7f1d1d30", "#7c2d1230"]
            c_color = badge_colors[sim_cluster]
            c_bg = badge_bg[sim_cluster]
            
            # Wrap indented multiline HTML in textwrap.dedent
            st.markdown(textwrap.dedent(f"""
            <div class="risk-banner {risk_style}">
                <div style="font-size: 1.05rem; font-weight: 800;">{risk_icon} Account Status: {risk_cat}</div>
                <div style="font-size: 1.85rem; font-weight: 900; margin: 0.3rem 0; line-height: 1.1;">{prob_churn:.1%}</div>
                <div style="font-size: 0.8rem; opacity: 0.95;">Probability of churn within next 30 days</div>
            </div>
            """), unsafe_allow_html=True)
            
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card" style="padding: 0.75rem;">
                    <div class="metric-val" style="font-size: 1.4rem; color: #f8fafc;">${sim_monthly:.2f}</div>
                    <div class="metric-label" style="font-size: 0.7rem;">Monthly Billing</div>
                </div>
                """), unsafe_allow_html=True)
            with sub_c2:
                st.markdown(textwrap.dedent(f"""
                <div class="metric-card" style="padding: 0.75rem; border: 1px solid #f9731640; background-color: #2c1a10;">
                    <div class="metric-val" style="font-size: 1.4rem; color: #fb923c;">${exp_loss_val:.2f}</div>
                    <div class="metric-label" style="font-size: 0.7rem;">Expected Monthly Loss</div>
                </div>
                """), unsafe_allow_html=True)
                
            st.markdown('<div class="section-title">Identified Segment Persona</div>', unsafe_allow_html=True)
            st.markdown(textwrap.dedent(f"""
            <div style="background-color: {c_bg}; border: 1px solid {c_color}60; border-radius: 10px; padding: 0.85rem;">
                <div class="persona-badge" style="background-color: {c_color}; color: white; font-size: 0.75rem; padding: 0.2rem 0.5rem;">Segment {sim_cluster}: {sim_profile['persona']}</div>
                <div style="font-size: 0.825rem; line-height: 1.4; color: #cbd5e1;">
                    <strong>Profile Description:</strong> {sim_profile['description']}<br>
                    <strong>Avg tenure in segment:</strong> {sim_profile['avg_tenure']:.1f} months &nbsp;|&nbsp; 
                    <strong>Avg Monthly Spend:</strong> ${sim_profile['avg_monthly_charges']:.2f}<br>
                    <strong>Segment Churn Rate:</strong> <span style="color: {c_color}; font-weight: bold;">{sim_profile['churn_rate']:.1%}</span>
                </div>
            </div>
            """), unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">Automated Retention Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = []
            
            if sim_contract == "Month-to-month":
                recommendations.append({
                    "title": "📜 Upgrade Contract Plan",
                    "desc": "Switching month-to-month contracts to 1-Year lock-in reduces churn rates by over 40% in this cohort. Propose a 12-month agreement with a 15% billing discount."
                })
                
            if "automatic" not in sim_payment:
                recommendations.append({
                    "title": "💳 Credit Card / Bank Auto-Pay Credit",
                    "desc": "Switching electronic or mailed check billing to Credit Card or Bank Transfer Auto-Pay drastically stabilizes tenure. Offer a one-time $10 account credit for sign-up."
                })
                
            if sim_internet == "Fiber optic" and (sim_security == "No" or sim_techsupport == "No"):
                recommendations.append({
                    "title": "🛡️ Bundle Premium Tech Support & Security",
                    "desc": "High-spend fiber users without technical services churn at high rates. Offer a 3-month free trial of Online Security + Tech Support to drive retention."
                })
                
            if sim_internet == "Fiber optic" and prob_churn > 0.5:
                recommendations.append({
                    "title": "📡 High-Performance Fiber Health Audit",
                    "desc": "Simulated user resides in a high-churn fiber segment. Task customer support to run a proactive remote speed diagnostic and contact client to check connection satisfaction."
                })
                
            if sim_tenure < 6 and prob_churn > 0.4:
                recommendations.append({
                    "title": "👋 New Account Relationship Check-in",
                    "desc": "Client is within critical early lifecycle (tenure < 6 mos) showing high churn risk. Flag account for a personalized outreach call from the Customer Success onboarding team."
                })
                
            if not recommendations:
                recommendations.append({
                    "title": "🎁 Loyalty Rewards & Premium Add-ons",
                    "desc": "This account exhibits stable behavior. Candidate is ripe for premium upsells (e.g. upgrading streaming plans or device protection programs) or enrollment in the VIP loyalty tier."
                })
                
            for rec in recommendations:
                st.markdown(textwrap.dedent(f"""
                <div class="recommendation-box">
                    <div class="rec-title">{rec['title']}</div>
                    <div class="rec-desc">{rec['desc']}</div>
                </div>
                """), unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: COHORTS & SEGMENTS
    # ---------------------------------------------------------
    with tab_cohorts:
        st.markdown('<div class="section-title">Customer Segmentation & Cohort Mapping (K-Means + PCA)</div>', unsafe_allow_html=True)
        
        coh_col_plot, coh_col_stats = st.columns([1.1, 1])
        
        with coh_col_plot:
            st.markdown("### 🗺️ PCA Dimension Space Projection")
            st.write("Visualizing the customer base in 2D space. K-Means clustering partitions customers based on demographics, billing, and services. The large yellow star represents the active customer simulated in Tab 2.")
            
            df_pca_plot = pd.DataFrame(df_raw.copy())
            
            encoded_all = preprocessor_clustering.transform(df_raw.drop(columns=['customerID', 'Churn']).assign(SeniorCitizen=df_raw['SeniorCitizen_Str']))
            all_clusters = kmeans_model.predict(encoded_all)
            all_pca = pca_model.transform(encoded_all)
            
            df_pca_plot['Cluster'] = all_clusters
            df_pca_plot['PCA_1'] = all_pca[:, 0]
            df_pca_plot['PCA_2'] = all_pca[:, 1]
            
            cluster_id_to_persona = {item['cluster_id']: item['persona'] for item in cluster_profiles}
            df_pca_plot['Persona'] = df_pca_plot['Cluster'].map(cluster_id_to_persona)
            
            fig_pca, ax_pca = plt.subplots(figsize=(6.5, 4.5))
            
            sns.scatterplot(
                data=df_pca_plot,
                x='PCA_1', 
                y='PCA_2',
                hue='Persona',
                palette=['#3b82f6', '#10b981', '#ef4444', '#fb923c'],
                alpha=0.3,
                s=12,
                edgecolor=None,
                ax=ax_pca
            )
            
            if 'sim_pca_coords' in locals():
                ax_pca.scatter(
                    sim_pca_coords[0], 
                    sim_pca_coords[1], 
                    color='#eab308', 
                    edgecolor='#0f172a',
                    s=200, 
                    marker='*', 
                    linewidths=1.5,
                    label='Simulated Customer',
                    zorder=10
                )
                
            ax_pca.set_title("Customer Cohorts Distribution Map", fontsize=11, fontweight='bold', color='#f1f5f9')
            ax_pca.set_xlabel("PCA Dimension 1 (Demographics & Service Density)", fontsize=8.5)
            ax_pca.set_ylabel("PCA Dimension 2 (Contract & Billing Mode)", fontsize=8.5)
            ax_pca.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5)
            
            apply_dark_plot_style(ax_pca, fig_pca)
            plt.tight_layout()
            st.pyplot(fig_pca)
            
        with coh_col_stats:
            st.markdown("### 📊 Cohort KPI Profiles")
            st.write("Summary statistics for each cluster derived during model training:")
            
            for profile in cluster_profiles:
                p_id = profile['cluster_id']
                p_name = profile['persona']
                p_desc = profile['description']
                p_size = profile['size']
                p_churn = profile['churn_rate']
                p_monthly = profile['avg_monthly_charges']
                p_tenure = profile['avg_tenure']
                
                colors = ["#3b82f6", "#10b981", "#ef4444", "#fb923c"]
                c_theme = colors[p_id]
                
                highlight_border = "border: 2px solid #eab308; box-shadow: 0 0 8px rgba(234, 179, 8, 0.25);" if ('sim_cluster' in locals() and sim_cluster == p_id) else "border: 1px solid #334155;"
                highlight_marker = "🌟 <span style='color: #eab308; font-weight: bold; font-size: 0.8rem;'>Simulated Customer Cluster</span><br>" if ('sim_cluster' in locals() and sim_cluster == p_id) else ""
                
                # Wrapped in textwrap.dedent to strip spaces and prevent code block rendering!
                st.markdown(textwrap.dedent(f"""
                <div style="background-color: #1e293b; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem; {highlight_border}">
                    {highlight_marker}
                    <div style="font-weight: 800; font-size: 0.925rem; color: {c_theme}; margin-bottom: 0.2rem;">Segment {p_id}: {p_name} ({p_size:,} Accounts)</div>
                    <div style="font-size: 0.775rem; color: #cbd5e1; font-style: italic; margin-bottom: 0.4rem; line-height: 1.3;">{p_desc}</div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.775rem; font-weight: 600; color: #cbd5e1;">
                        <span>Tenure: {p_tenure:.1f} mo</span>
                        <span>Monthly: ${p_monthly:.2f}</span>
                        <span style="color: {'#ef4444' if p_churn > 0.25 else '#34d399'};">Churn: {p_churn:.1%}</span>
                    </div>
                </div>
                """), unsafe_allow_html=True)
                
    # ---------------------------------------------------------
    # TAB 4: BATCH ANALYSIS & EXPORT
    # ---------------------------------------------------------
    with tab_batch:
        st.markdown('<div class="section-title">Batch Analytics & Retention Outreach Target Generator</div>', unsafe_allow_html=True)
        st.write("Generate and export lists of high-value customers with elevated churn risks to power targeted retention outreach programs.")
        
        df_batch = df_raw.copy()
        
        df_batch['ChurnProbability'] = churn_probs
        df_batch['ExpectedMonthlyRevenueLoss'] = df_batch['ChurnProbability'] * df_batch['MonthlyCharges']
        
        encoded_all = preprocessor_clustering.transform(df_raw.drop(columns=['customerID', 'Churn']).assign(SeniorCitizen=df_raw['SeniorCitizen_Str']))
        df_batch['ClusterID'] = kmeans_model.predict(encoded_all)
        df_batch['CohortPersona'] = df_batch['ClusterID'].map(cluster_id_to_persona)
        
        df_batch_sorted = df_batch.sort_values(by='ExpectedMonthlyRevenueLoss', ascending=False)
        
        cutoff_slider = st.slider("Select Churn Probability Cutoff (%) for outreach", 10, 90, 50)
        
        filtered_batch = df_batch_sorted[df_batch_sorted['ChurnProbability'] >= (cutoff_slider / 100.0)]
        
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            st.metric("Outreach Targets Found", f"{len(filtered_batch):,}")
        with b_c2:
            st.metric("Total Monthly Value at Risk", f"${filtered_batch['MonthlyCharges'].sum():,.2f}")
        with b_c3:
            st.metric("Total Expected Revenue Loss", f"${filtered_batch['ExpectedMonthlyRevenueLoss'].sum():,.2f}")
            
        display_cols = [
            'customerID', 'gender', 'tenure', 'Contract', 'InternetService', 
            'MonthlyCharges', 'ChurnProbability', 'ExpectedMonthlyRevenueLoss', 'CohortPersona'
        ]
        
        st.markdown("### 📋 Outreach Target List (Top 100 Highest Expected Loss)")
        
        df_display = filtered_batch[display_cols].head(100).copy()
        df_display['ChurnProbability'] = df_display['ChurnProbability'].map(lambda x: f"{x:.1%}")
        df_display['MonthlyCharges'] = df_display['MonthlyCharges'].map(lambda x: f"${x:.2f}")
        df_display['ExpectedMonthlyRevenueLoss'] = df_display['ExpectedMonthlyRevenueLoss'].map(lambda x: f"${x:.2f}")
        
        st.dataframe(df_display, use_container_width=True)
        
        export_df = filtered_batch[[
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
            'ChurnProbability', 'ExpectedMonthlyRevenueLoss', 'CohortPersona'
        ]]
        
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Export Complete Target List to CSV",
            data=csv_data,
            file_name=f"telecom_churn_outreach_targets_{cutoff_slider}pct.csv",
            mime="text/csv"
        )
        
else:
    st.info("System is offline. Please resolve the model training errors above to initialize the platform.")

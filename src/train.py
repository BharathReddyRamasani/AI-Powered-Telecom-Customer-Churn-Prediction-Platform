import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_serialize():
    print("🚀 Starting Model Training Pipeline...")
    
    # 1. Load Data
    # Use relative paths for portability
    data_path = 'data/Telco-Customer-Churn.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # 2. Data Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    print(f"Found {missing_count} rows with missing TotalCharges. Dropping them.")
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert SeniorCitizen to string category so it can be handled by OneHotEncoder easily
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Define features and target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Define numeric and categorical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    # Save column specifications for app reference
    col_specs = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'all_cols': list(X.columns)
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/col_specs.json', 'w') as f:
        json.dump(col_specs, f, indent=4)
        
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    # 3. Train-Test Split for Classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # 5. Classifier Training (Random Forest)
    # class_weight='balanced' handles target imbalance (26% churned vs 74% non-churned)
    clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=12,
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    classification_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    classification_pipeline.fit(X_train, y_train)
    print("Classification pipeline trained.")
    
    # Evaluate Classifier
    y_pred = classification_pipeline.predict(X_test)
    y_prob = classification_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\n=== Classifier Performance on Test Set ===")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
        
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save the classification pipeline
    joblib.dump(classification_pipeline, 'models/churn_model.pkl')
    print("Saved churn classifier pipeline to models/churn_model.pkl")
    
    # 6. Customer Clustering Pipeline
    print("\n👥 Running Customer Clustering (Segmentation)...")
    
    # We will fit the preprocessor on the entire X dataset for clustering purposes
    clustering_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    X_encoded = clustering_preprocessor.fit_transform(X)
    
    # Run K-Means with K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_encoded)
    
    # Fit PCA for 2D visualization in Streamlit
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_encoded)
    
    # Add clusters and PCA projections to a summary df for analysis
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    df_clustered['PCA_1'] = X_pca[:, 0]
    df_clustered['PCA_2'] = X_pca[:, 1]
    df_clustered['Churn_numeric'] = df_clustered['Churn'].map({'Yes': 1, 'No': 0})
    
    # Analyze clusters to define personas
    print("\n=== Cluster Characteristics Summary ===")
    cluster_profiles = []
    for c in range(4):
        c_data = df_clustered[df_clustered['Cluster'] == c]
        avg_tenure = c_data['tenure'].mean()
        avg_monthly = c_data['MonthlyCharges'].mean()
        avg_total = c_data['TotalCharges'].mean()
        churn_rate = c_data['Churn_numeric'].mean()
        
        # Determine dominant contract type
        dom_contract = c_data['Contract'].value_counts().index[0]
        # Determine dominant payment method
        dom_payment = c_data['PaymentMethod'].value_counts().index[0]
        # Determine internet service distribution
        dom_internet = c_data['InternetService'].value_counts().index[0]
        
        # Assign a Persona
        if dom_contract in ['One year', 'Two year'] and avg_monthly >= 65:
            persona = "Premium Loyalists"
            desc = "Long-term contract, high monthly spend. High value, low risk. Value premium services, support, and stability."
        elif dom_contract == 'Month-to-month' and avg_monthly >= 65:
            persona = "High-Risk Tech-Savvy"
            desc = "Month-to-month, high monthly charges. Heavy internet/streaming users, highly sensitive to service quality and price. High risk of churn."
        elif dom_contract == 'Month-to-month' and avg_monthly < 65:
            persona = "Budget Month-to-Month"
            desc = "Month-to-month, low monthly charges. Mostly basic services (phone only or DSL). High price sensitivity, moderate-to-high risk."
        else:
            persona = "Standard Savers"
            desc = "Long-term contract, low monthly spend. Budget-conscious but stable, highly loyal, low churn risk."
            
        profile = {
            'cluster_id': int(c),
            'persona': persona,
            'description': desc,
            'avg_tenure': float(avg_tenure),
            'avg_monthly_charges': float(avg_monthly),
            'avg_total_charges': float(avg_total),
            'churn_rate': float(churn_rate),
            'dominant_contract': dom_contract,
            'dominant_payment': dom_payment,
            'dominant_internet': dom_internet,
            'size': int(len(c_data))
        }
        cluster_profiles.append(profile)
        
        print(f"Cluster {c} ({persona}): size={len(c_data)}, tenure={avg_tenure:.1f} mo, monthly_charge=${avg_monthly:.2f}, churn={churn_rate:.1%}")
        print(f"  Dominant Contract: {dom_contract}, Internet: {dom_internet}")
    
    # Save clustering models and information
    joblib.dump(clustering_preprocessor, 'models/preprocessor_for_clustering.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(pca, 'models/pca_model.pkl')
    
    with open('models/cluster_profiles.json', 'w') as f:
        json.dump(cluster_profiles, f, indent=4)
        
    print("Saved clustering artifacts successfully.")
    print("🎉 Model training completed successfully!")

if __name__ == '__main__':
    train_and_serialize()

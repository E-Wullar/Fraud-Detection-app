# Fraud_detection.py (FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Set page configuration
st.set_page_config(
    page_title="BrightHorizon Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 BrightHorizon Fraud Detection Dashboard")
st.write("Real-time fraud detection for financial transactions")

# Check if model files exist
model_exists = os.path.exists('fraud_detection_model.pkl')
feature_importance_exists = os.path.exists('feature_importance.csv')
performance_exists = os.path.exists('model_performance.csv')

if not model_exists or not feature_importance_exists:
    st.error("❌ Model files not found! Please train the model first.")
    if os.path.exists('.'):
        st.info("Current files in folder: " + ", ".join(os.listdir('.')))
    st.stop()

# Load model, feature importance, and performance data
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        feature_importance = pd.read_csv('feature_importance.csv')
        performance_data = pd.read_csv('model_performance.csv') if performance_exists else None
        return model, feature_importance, performance_data
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, feature_importance, performance_data = load_assets()

if model is None:
    st.stop()

# Get the actual features used by the model
@st.cache_data
def get_model_features():
    # Extract feature names from the model (if available)
    try:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            # Fallback: use the features from feature importance file
            features = feature_importance['feature'].tolist()
        return features
    except:
        return []

# Get all features the model expects
all_model_features = get_model_features()
st.sidebar.info(f"Model expects {len(all_model_features)} features")

# Get features to display to user (excluding Is_Night and Is_Business_Hours)
display_features = [f for f in all_model_features if f not in ['Is_Night', 'Is_Business_Hours']]

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["Feature Importance", "Fraud Prediction", "Model Performance"]
page = st.sidebar.radio("Go to", pages)

if page == "Feature Importance":
    st.header("📊 Feature Importance")
    
    # Display top features (excluding Is_Night and Is_Business_Hours)
    top_features = feature_importance[~feature_importance['feature'].isin(['Is_Night', 'Is_Business_Hours'])].head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
    ax.set_title('Features Importance for Fraud Detection')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("Important Features:")
    st.dataframe(top_features.head(10))
    
elif page == "Fraud Prediction":
    st.header("🔮 Fraud Prediction")
    
    # Create input form with the features to display to user
    with st.form("prediction_form"):
        st.subheader("Transaction Details")
        st.info(f"Please provide values for the {len(display_features)} features used by the model")
        
        input_values = {}
        
        # Create input fields for each feature to display to user
        for feature in display_features:
            if feature == 'Absolute_Amount':
                input_values[feature] = st.number_input("Absolute Amount", min_value=0.0, value=1000.0, step=100.0)
            elif feature == 'Is_Debit' or feature == 'Is_Credit':
                # Skip these as we'll handle them with a combined dropdown
                continue
            elif feature == 'Is_Round_Amount':
                input_values[feature] = st.selectbox("Is Round Amount?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            elif feature == 'Transaction_Hour':
                input_values[feature] = st.slider("Transaction Hour", 0, 23, 12)
            elif feature == 'Is_Weekend':
                input_values[feature] = st.selectbox("Is Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            elif feature == 'Balance_Change':
                input_values[feature] = st.number_input("Balance Change", value=0.0, step=100.0)
            elif feature == 'Is_Mobile_Channel' or feature == 'Is_ATM_Channel' or feature == 'Is_Branch_Channel':
                # Skip these as we'll handle them with a combined dropdown
                continue
            elif feature == 'Current_Balance':
                input_values[feature] = st.number_input("Current Balance", min_value=0.0, value=5000.0, step=100.0)
            elif feature == 'Account_Age_Years':
                input_values[feature] = st.number_input("Account Age (Years)", min_value=0.0, value=2.5, step=0.5)
            elif feature == 'Days_Since_Last_Txn':
                input_values[feature] = st.number_input("Days Since Last Transaction", min_value=0, value=7, step=1)
            elif feature == 'Loan_Disbursement_Amount':
                input_values[feature] = st.number_input("Loan Disbursement Amount", min_value=0.0, value=0.0, step=1000.0)
            elif feature == 'Loan_Interest_Rate_(%)':
                input_values[feature] = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            elif feature == 'Loan_Loan_Cycle':
                input_values[feature] = st.number_input("Loan Cycle", min_value=0, value=0, step=1)
            elif feature == 'NPL_Rate_(%)':
                input_values[feature] = st.number_input("NPL Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            elif feature == 'Staff_Count':
                input_values[feature] = st.number_input("Staff Count", min_value=0, value=10, step=1)
            else:
                # Default input for any unexpected features
                input_values[feature] = st.number_input(f"{feature}", value=0.0)
        
        # Add combined transaction type dropdown
        transaction_type = st.selectbox("Transaction Type", options=["Debit", "Credit"])
        
        # Add combined channel dropdown
        channel = st.selectbox("Transaction Channel", options=["Mobile", "ATM", "Branch"])
        
        submitted = st.form_submit_button("🔍 Predict Fraud Risk")
    
    if submitted:
        try:
            # Set debit/credit values based on dropdown selection
            input_values['Is_Debit'] = 1 if transaction_type == "Debit" else 0
            input_values['Is_Credit'] = 1 if transaction_type == "Credit" else 0
            
            # Set channel values based on dropdown selection
            input_values['Is_Mobile_Channel'] = 1 if channel == "Mobile" else 0
            input_values['Is_ATM_Channel'] = 1 if channel == "ATM" else 0
            input_values['Is_Branch_Channel'] = 1 if channel == "Branch" else 0
            
            # Set default values for excluded features
            input_values['Is_Night'] = 0
            input_values['Is_Business_Hours'] = 0
            
            # Prepare input data in the EXACT order the model expects
            input_array = np.array([[input_values[feature] for feature in all_model_features]])
            
            # Make prediction
            probability = model.predict_proba(input_array)[0][1]
            
            # Display results
            st.subheader("Prediction Results")
            
            if probability > 0.7:
                st.error(f"🚨 HIGH FRAUD RISK: {probability:.2%}")
                st.info("This transaction shows strong indicators of potential fraud. Recommend manual review.")
            elif probability > 0.3:
                st.warning(f"⚠️ MEDIUM RISK: {probability:.2%}")
                st.info("This transaction requires additional verification.")
            else:
                st.success(f"✅ LOW RISK: {probability:.2%}")
                st.info("This transaction appears to be legitimate.")
            
            # Show feature values for debugging
            with st.expander("View Feature Values Used"):
                feature_df = pd.DataFrame({
                    'Feature': all_model_features,
                    'Value': [input_values[feature] for feature in all_model_features]
                })
                st.dataframe(feature_df)
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("This usually happens when the input data doesn't match what the model was trained on.")
            st.info(f"Model expects {len(all_model_features)} features: {', '.join(all_model_features)}")

elif page == "Model Performance":
    st.header("📈 Model Performance Report")
    
    if performance_data is not None:
        # Display performance metrics
        st.subheader("Performance Metrics")
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if 'accuracy' in performance_data.columns:
                st.metric("Accuracy", f"{performance_data['accuracy'].iloc[0]:.2%}")
            if 'precision' in performance_data.columns:
                st.metric("Precision", f"{performance_data['precision'].iloc[0]:.2%}")
            if 'recall' in performance_data.columns:
                st.metric("Recall", f"{performance_data['recall'].iloc[0]:.2%}")
        
        with col2:
            if 'f1_score' in performance_data.columns:
                st.metric("F1 Score", f"{performance_data['f1_score'].iloc[0]:.2%}")
            if 'roc_auc' in performance_data.columns:
                st.metric("ROC AUC", f"{performance_data['roc_auc'].iloc[0]:.2%}")
        
        # Display confusion matrix if available
        if all(col in performance_data.columns for col in ['tn', 'fp', 'fn', 'tp']):
            st.subheader("Confusion Matrix")
            tn = performance_data['tn'].iloc[0]
            fp = performance_data['fp'].iloc[0]
            fn = performance_data['fn'].iloc[0]
            tp = performance_data['tp'].iloc[0]
            
            cm = np.array([[tn, fp], [fn, tp]])
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        # Display classification report if available
        if 'classification_report' in performance_data.columns:
            st.subheader("Classification Report")
            st.text(performance_data['classification_report'].iloc[0])
    else:
        st.warning("No performance data available. Please generate model performance metrics.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("BrightHorizon Fraud Detection System")
st.sidebar.info(f"Model features: {len(all_model_features)}")

# Debug information
with st.sidebar.expander("Debug Info"):
    st.write(f"Model features: {all_model_features}")
    st.write(f"Number of features: {len(all_model_features)}")
    if hasattr(model, 'n_features_in_'):
        st.write(f"Model expects: {model.n_features_in_} features")
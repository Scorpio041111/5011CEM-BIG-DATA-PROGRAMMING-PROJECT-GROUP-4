import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import time

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Analytics Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and animations
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load the heart disease dataset with comprehensive error handling"""
    excel_path = r"C:\Users\Xiang\20250627_Big_Data\20250630_Heart_2022_V3.xlsx"
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: `{excel_path}`")
        return pd.DataFrame()
    except ImportError:
        st.error("❌ `openpyxl` is not installed. Please install it using `pip install openpyxl`.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error loading Excel file: {e}")
        return pd.DataFrame()


def data_preprocessing(df):
    """Advanced data preprocessing with multiple imputation strategies"""
    df_processed = df.copy()
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
    
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    return df_processed, label_encoders

def create_overview_metrics(df):
    """Create overview metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df.columns)}</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>{missing_percentage:.1f}%</h3>
            <p>Missing Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'HadHeartAttack' in df.columns:
            heart_attack_rate = (df['HadHeartAttack'].sum() / len(df)) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>{heart_attack_rate:.1f}%</h3>
                <p>Heart Attack Rate</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>N/A</h3>
                <p>Heart Attack Rate</p>
            </div>
            """, unsafe_allow_html=True)

def create_interactive_correlation_heatmap(df):
    """Create an interactive correlation heatmap using Plotly"""
    st.markdown('<div class="sub-header">🔥 Interactive Correlation Heatmap</div>', unsafe_allow_html=True)
    
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        st.warning("Not enough numeric columns for correlation analysis.")
        return
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        width=800,
        height=600,
        title_x=0.5,
        transition_duration=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_distribution_analysis(df):
    """Create comprehensive distribution analysis"""
    st.markdown('<div class="sub-header">📊 Feature Distribution Analysis</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for distribution analysis.")
        return
    
    selected_features = st.multiselect(
        "Select features to analyze:",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    cols = 2
    rows = (len(selected_features) + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=selected_features
    )
    
    for i, feature in enumerate(selected_features):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[feature].dropna(),
                name=f'{feature} Distribution',
                opacity=0.7,
                nbinsx=30
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Feature Distributions",
        height=300 * rows,
        showlegend=False,
        transition_duration=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled_data=False, scaler=None):
    """Helper function to train a single model and return its metrics."""
    start_time = time.time()
    
    X_train_data = scaler.transform(X_train) if use_scaled_data else X_train
    X_test_data = scaler.transform(X_test) if use_scaled_data else X_test
    
    model.fit(X_train_data, y_train)
    
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    
    end_time = time.time()
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    training_time = (end_time - start_time) / 60  # in minutes

    return {
        'model': model,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score,
        'training_time_min': training_time
    }

def display_model_results(model_name, results):
    """Displays the results for a single model."""
    st.markdown(f'<div class="sub-header">⚙️ {model_name} Performance</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{results['accuracy']:.2%}")
    col2.metric("AUC Score", f"{results['auc']:.4f}")
    col3.metric("Training Time (min)", f"{results['training_time_min']:.4f}")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        fig = px.imshow(results['confusion_matrix'], text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No Heart Attack', 'Heart Attack'],
                        y=['No Heart Attack', 'Heart Attack'],
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['fpr'], y=results['tpr'], mode='lines', name=f"ROC (AUC={results['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(results['report']).transpose()
    st.dataframe(report_df)

def main():
    # Header
    st.markdown('<div class="main-header">❤️ Heart Disease Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Team Members:** Arief bin Abdul Latib, Khor Cojean, Lim Yong Xiang, Ooi Yong Hang")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df_raw = load_data()
        if df_raw.empty:
            st.stop()
    
    # Data preprocessing
    with st.spinner("Preprocessing data..."):
        df_clean, label_encoders = data_preprocessing(df_raw)
    
    # Sidebar options
    st.sidebar.subheader("Select Analysis")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ["📊 Data Overview", "🔍 Exploratory Analysis"]
    )

    st.sidebar.subheader("Machine Learning Models")
    model_selection = st.sidebar.radio(
        "Choose a model or comparison:",
        ["None", "Linear Regression", "Random Forest", "Support Vector Machine", "Neural Network", "📈 Compare All Models"]
    )

    # Main content based on selection
    if analysis_type == "📊 Data Overview":
        st.header("📊 Data Overview")
        create_overview_metrics(df_clean)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Sample")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        with col2:
            st.subheader("Processed Data Sample")
            st.dataframe(df_clean.head(), use_container_width=True)
    
    elif analysis_type == "🔍 Exploratory Analysis":
        st.header("🔍 Exploratory Data Analysis")
        create_interactive_correlation_heatmap(df_clean)
        create_distribution_analysis(df_clean)

    # Machine Learning Section
    if model_selection != "None":
        st.header("🤖 Machine Learning Analysis")
        
        target_col = 'HadHeartAttack'
        numeric_df = df_clean.select_dtypes(include=[np.number])
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        scaler = StandardScaler().fit(X_train)

        models = {
            'Linear Regression': (LinearRegression(), True), # Note: This is not ideal for classification
            'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
            'Support Vector Machine': (SVC(probability=True, random_state=42), True),
            'Neural Network': (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), True)
        }

        if model_selection in models:
            with st.spinner(f"Training {model_selection}..."):
                model_obj, use_scaled = models[model_selection]
                # A special case for Linear Regression to make it a classifier
                if model_selection == 'Linear Regression':
                    start_time = time.time()
                    model_obj.fit(scaler.transform(X_train), y_train)
                    y_pred_continuous = model_obj.predict(scaler.transform(X_test))
                    y_pred = (y_pred_continuous > 0.5).astype(int)
                    y_pred_proba = np.clip(y_pred_continuous, 0, 1)
                    end_time = time.time()
                    
                    results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'fpr': roc_curve(y_test, y_pred_proba)[0],
                        'tpr': roc_curve(y_test, y_pred_proba)[1],
                        'auc': auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]),
                        'training_time_min': (end_time - start_time) / 60
                    }
                else:
                    results = train_and_evaluate_model(model_obj, X_train, y_train, X_test, y_test, use_scaled_data=use_scaled, scaler=scaler)
                
                display_model_results(model_selection, results)

        elif model_selection == "📈 Compare All Models":
            st.markdown('<div class="sub-header">📊 Model Comparison and Analysis</div>', unsafe_allow_html=True)
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, (model, use_scaled)) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                if name == 'Linear Regression':
                    start_time = time.time()
                    model.fit(scaler.transform(X_train), y_train)
                    y_pred_continuous = model.predict(scaler.transform(X_test))
                    y_pred_proba = np.clip(y_pred_continuous, 0, 1)
                    end_time = time.time()
                    auc_score_val = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])
                    accuracy_val = accuracy_score(y_test, (y_pred_continuous > 0.5).astype(int))
                    time_val = (end_time - start_time) / 60
                else:
                    res = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, use_scaled_data=use_scaled, scaler=scaler)
                    accuracy_val = res['accuracy']
                    auc_score_val = res['auc']
                    time_val = res['training_time_min']

                all_results.append({
                    'Model': name,
                    'Accuracy': accuracy_val,
                    'AUC': auc_score_val,
                    'Training Time (min)': time_val
                })
                progress_bar.progress((i + 1) / len(models))
            
            status_text.success("All models trained!")
            progress_bar.empty()

            results_df = pd.DataFrame(all_results).sort_values('AUC', ascending=False)
            st.dataframe(results_df.style.format({
                'Accuracy': '{:.2%}',
                'AUC': '{:.4f}',
                'Training Time (min)': '{:.4f}'
            }).background_gradient(cmap='viridis', subset=['Accuracy', 'AUC']), use_container_width=True)

            st.markdown("---")
            st.subheader("Performance Visualization")
            fig = px.bar(results_df, x='Model', y=['Accuracy', 'AUC'], barmode='group',
                         title="Model Accuracy and AUC Comparison")
            st.plotly_chart(fig, use_container_width=True)


    # Footer
    st.markdown("---")
    st.markdown("*5011CEM Big Data Programming*")

if __name__ == "__main__":
    main()
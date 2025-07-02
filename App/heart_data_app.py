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
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Analytics Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    # Handle mixed data types and convert to appropriate types
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Try to convert to numeric first
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='ignore')
            except:
                pass
    
    # Separate numeric and categorical columns after type conversion
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    # Impute numeric columns with median (more robust than mean)
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = numeric_imputer.fit_transform(df_processed[numeric_cols])
    
    # Impute categorical columns with mode
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = categorical_imputer.fit_transform(df_processed[categorical_cols])
    
    # Encode categorical variables
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
    
    # Create interactive heatmap
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
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("**🎯 Strongest Correlations:**")
    for i, (feat1, feat2, corr) in enumerate(corr_pairs[:5]):
        if not np.isnan(corr):
            st.write(f"{i+1}. {feat1} ↔ {feat2}: {corr:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

def create_distribution_analysis(df):
    """Create comprehensive distribution analysis"""
    st.markdown('<div class="sub-header">📊 Feature Distribution Analysis</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for distribution analysis.")
        return
    
    # Feature selection
    selected_features = st.multiselect(
        "Select features to analyze:",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
    )
    
    if not selected_features:
        st.warning("Please select at least one feature.")
        return
    
    # Create subplots
    cols = 2
    rows = (len(selected_features) + 1) // 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=selected_features,
        specs=[[{"secondary_y": True}] * cols for _ in range(rows)]
    )
    
    for i, feature in enumerate(selected_features):
        row = i // cols + 1
        col = i % cols + 1
        
        # Histogram
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
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_target_analysis(df, target_col='HadHeartAttack'):
    """Analyze target variable relationships"""
    if target_col not in df.columns:
        st.warning(f"Target column '{target_col}' not found.")
        return
    
    st.markdown('<div class="sub-header">🎯 Heart Attack Risk Analysis</div>', unsafe_allow_html=True)
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        target_counts = df[target_col].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=['No Heart Attack', 'Heart Attack'],
            title="Heart Attack Distribution",
            color_discrete_sequence=['#90EE90', '#FF6B6B']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age group analysis
        if 'Age' in df.columns:
            # Ensure Age is numeric and handle any potential string values
            df_age = df.copy()
            df_age['Age'] = pd.to_numeric(df_age['Age'], errors='coerce')
            df_age = df_age.dropna(subset=['Age'])
            
            if len(df_age) > 0:
                # Create age groups
                df_age['AgeGroup'] = pd.cut(
                    df_age['Age'], 
                    bins=[0, 30, 50, 70, 100], 
                    labels=['<30', '30-50', '50-70', '70+'],
                    include_lowest=True
                )
                
                # Calculate heart attack rate by age group
                age_heart_attack = df_age.groupby('AgeGroup')[target_col].mean().reset_index()
                
                fig = px.bar(
                    age_heart_attack,
                    x='AgeGroup',
                    y=target_col,
                    title="Heart Attack Rate by Age Group",
                    color=target_col,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis_title="Heart Attack Rate")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid age data found for analysis.")
        else:
            st.warning("Age column not found for age group analysis.")

def create_risk_factor_analysis(df):
    """Analyze various risk factors"""
    st.markdown('<div class="sub-header">⚠️ Risk Factor Analysis</div>', unsafe_allow_html=True)
    
    risk_factors = ['Smoking', 'AlcoholDrinking', 'PhysicalActivity', 'Diabetic', 'HadStroke']
    available_factors = [factor for factor in risk_factors if factor in df.columns]
    
    if not available_factors:
        st.warning("No risk factor columns found.")
        return
    
    if 'HadHeartAttack' not in df.columns:
        st.warning("Target variable 'HadHeartAttack' not found.")
        return
    
    # Calculate risk ratios
    risk_data = []
    for factor in available_factors:
        if factor in df.columns:
            risk_ratio = df.groupby(factor)['HadHeartAttack'].mean()
            if len(risk_ratio) >= 2:
                ratio = risk_ratio.iloc[1] / risk_ratio.iloc[0] if risk_ratio.iloc[0] > 0 else 0
                risk_data.append({
                    'Risk Factor': factor,
                    'Risk Ratio': ratio,
                    'Baseline Rate': risk_ratio.iloc[0],
                    'Risk Rate': risk_ratio.iloc[1] if len(risk_ratio) > 1 else 0
                })
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.bar(
            risk_df,
            x='Risk Factor',
            y='Risk Ratio',
            title="Risk Ratios for Heart Attack",
            color='Risk Ratio',
            color_continuous_scale='Reds'
        )
        fig.add_hline(y=1, line_dash="dash", line_color="black", annotation_text="Baseline Risk")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.write("**📈 Risk Factor Insights:**")
        for _, row in risk_df.iterrows():
            if row['Risk Ratio'] > 1.2:
                st.write(f"• {row['Risk Factor']}: {row['Risk Ratio']:.2f}x higher risk")
        st.markdown('</div>', unsafe_allow_html=True)

def train_multiple_models(df, target_col='HadHeartAttack'):
    """Train and compare Linear Regression, Random Forest, SVM, and Neural Network models"""
    st.markdown('<div class="sub-header">🤖 Machine Learning Model Comparison</div>', unsafe_allow_html=True)
    
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        st.warning(f"Target column '{target_col}' not found in numeric data.")
        return
    
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.warning("No valid data available for model training.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Train and evaluate models
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        progress_bar.progress((i + 1) / len(models))
        
        try:
            if name == 'Linear Regression':
                # For linear regression, we'll use it as a classifier by thresholding
                model.fit(X_train_scaled, y_train)
                y_pred_continuous = model.predict(X_test_scaled)
                y_pred = (y_pred_continuous > 0.5).astype(int)
                y_pred_proba = np.clip(y_pred_continuous, 0, 1)
                
                accuracy = accuracy_score(y_test, y_pred)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
                
            else:
                # For classification models
                if name in ['Support Vector Machine', 'Neural Network']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'AUC': auc_score,
                'FPR': fpr,
                'TPR': tpr,
                'Trained': True
            })
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            results.append({
                'Model': name,
                'Accuracy': 0,
                'AUC': 0,
                'FPR': [0, 1],
                'TPR': [0, 1],
                'Trained': False
            })
    
    progress_bar.empty()
    status_text.empty()
    
    # Filter out failed models
    successful_results = [r for r in results if r['Trained']]
    
    if not successful_results:
        st.error("No models trained successfully.")
        return
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        results_df = pd.DataFrame(successful_results)[['Model', 'Accuracy', 'AUC']]
        fig = px.bar(
            results_df,
            x='Model',
            y='Accuracy',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC curves
        fig = go.Figure()
        for result in successful_results:
            fig.add_trace(go.Scatter(
                x=result['FPR'],
                y=result['TPR'],
                mode='lines',
                name=f"{result['Model']} (AUC: {result['AUC']:.3f})",
                line=dict(width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.subheader("📊 Model Performance Summary")
    performance_df = pd.DataFrame(successful_results)[['Model', 'Accuracy', 'AUC']].round(4)
    performance_df = performance_df.sort_values('AUC', ascending=False)
    st.dataframe(performance_df, use_container_width=True)
    
    # Best model analysis
    if successful_results:
        best_model_idx = max(range(len(successful_results)), key=lambda i: successful_results[i]['AUC'])
        best_model_name = successful_results[best_model_idx]['Model']
        
        st.markdown(f'<div class="insight-box">', unsafe_allow_html=True)
        st.write(f"**🏆 Best Performing Model: {best_model_name}**")
        st.write(f"• Accuracy: {successful_results[best_model_idx]['Accuracy']:.4f}")
        st.write(f"• AUC Score: {successful_results[best_model_idx]['AUC']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance for Random Forest
        if best_model_name == 'Random Forest':
            best_model = models[best_model_name]
            feature_importance = pd.Series(
                best_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=feature_importance.values[:10],
                y=feature_importance.index[:10],
                orientation='h',
                title=f"Top 10 Feature Importances - {best_model_name}",
                labels={'x': 'Importance', 'y': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

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
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📊 Data Overview", "🔍 Exploratory Analysis", "🎯 Risk Analysis", "🤖 Machine Learning", "📈 All Analyses"]
    )
    
    # Main content based on selection
    if analysis_type == "📊 Data Overview" or analysis_type == "📈 All Analyses":
        st.header("📊 Data Overview")
        create_overview_metrics(df_clean)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Sample")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        with col2:
            st.subheader("Processed Data Sample")
            st.dataframe(df_clean.head(), use_container_width=True)
        
        # Data quality analysis
        st.subheader("📋 Data Quality Report")
        missing_data = df_raw.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Missing Values by Feature",
                labels={'x': 'Missing Count', 'y': 'Features'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    if analysis_type == "🔍 Exploratory Analysis" or analysis_type == "📈 All Analyses":
        st.header("🔍 Exploratory Data Analysis")
        create_interactive_correlation_heatmap(df_clean)
        create_distribution_analysis(df_clean)
    
    if analysis_type == "🎯 Risk Analysis" or analysis_type == "📈 All Analyses":
        st.header("🎯 Risk Factor Analysis")
        create_target_analysis(df_clean)
        create_risk_factor_analysis(df_clean)
    
    if analysis_type == "🤖 Machine Learning" or analysis_type == "📈 All Analyses":
        st.header("🤖 Machine Learning Analysis")
        train_multiple_models(df_clean)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with ❤️ using Streamlit and Plotly*")

if __name__ == "__main__":
    main()
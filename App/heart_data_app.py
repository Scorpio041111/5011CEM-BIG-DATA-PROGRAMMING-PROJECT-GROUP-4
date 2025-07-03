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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, mean_squared_error, r2_score, precision_score, recall_score, f1_score
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
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .execution-time {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    .reference-guide {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .reference-header {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 0.5rem;
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

def create_quick_reference_guide(df_original, df_processed, label_encoders):
    """Create a quick reference guide for encoded values"""
    st.markdown('<div class="sub-header">📚 Quick Reference Guide</div>', unsafe_allow_html=True)
    
    if not label_encoders:
        st.info("No categorical variables were encoded in this dataset.")
        return
    
    st.markdown("""
    <div class="reference-guide">
        <div class="reference-header">🔍 Encoded Categories Mapping</div>
        <p>This guide shows how categorical values were converted to numbers for analysis:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    num_cols = min(2, len(label_encoders))
    cols = st.columns(num_cols)
    
    for idx, (column, encoder) in enumerate(label_encoders.items()):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            st.markdown(f"""
            <div class="reference-guide">
                <div class="reference-header">📊 {column}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get original values and their encoded counterparts
            original_values = df_original[column].dropna().unique()
            
            # Create mapping dataframe
            mapping_data = []
            for orig_val in original_values:
                try:
                    encoded_val = encoder.transform([str(orig_val)])[0]
                    mapping_data.append({
                        'Original Value': orig_val,
                        'Encoded Value': encoded_val
                    })
                except:
                    continue
            
            if mapping_data:
                mapping_df = pd.DataFrame(mapping_data)
                mapping_df = mapping_df.sort_values('Encoded Value')
                
                # Display as a clean table
                st.dataframe(
                    mapping_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add summary statistics
                st.markdown(f"""
                <small>
                📈 <strong>Summary:</strong> {len(mapping_df)} unique values encoded<br>
                🔢 <strong>Range:</strong> {mapping_df['Encoded Value'].min()} to {mapping_df['Encoded Value'].max()}
                </small>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"No mapping available for {column}")
    
    # Add interpretation guide
    st.markdown("""
    <div class="reference-guide">
        <div class="reference-header">💡 How to Use This Guide</div>
        <ul>
            <li><strong>Original Value:</strong> The actual category name from your data</li>
            <li><strong>Encoded Value:</strong> The number used in analysis and machine learning</li>
            <li><strong>Remember:</strong> Higher encoded values don't mean "better", they're just labels</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def prepare_ml_data(df, target_col='HadHeartAttack'):
    """Prepare data for machine learning with consistent train-test split"""
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        st.error(f"Target column '{target_col}' not found in numeric data.")
        return None, None, None, None, None, None
    
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.error("No valid data available for model training.")
        return None, None, None, None, None, None
    
    # Split data with consistent parameters
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    return X, y, X_train, X_test, y_train, y_test

def calculate_model_metrics(y_test, y_pred, y_pred_proba):
    """Calculate comprehensive model metrics"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr
    }

def display_model_results(model_name, metrics, execution_time, y_test, y_pred, feature_names=None, feature_importance=None):
    """Display comprehensive model results"""
    
    # Execution time display
    st.markdown(f"""
    <div class="execution-time">
        ⏱️ <strong>{model_name} Execution Time:</strong> {execution_time:.4f} minutes ({execution_time*60:.2f} seconds)
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['accuracy']:.4f}</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['precision']:.4f}</h3>
            <p>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['recall']:.4f}</h3>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics['f1_score']:.4f}</h3>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ROC Curve and Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metrics['fpr'],
            y=metrics['tpr'],
            mode='lines',
            name=f"{model_name} (AUC: {metrics['auc']:.4f})",
            line=dict(width=3, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Random Classifier'
        ))
        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=400,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix - {model_name}",
            labels=dict(x="Predicted", y="Actual"),
            x=['No Heart Attack', 'Heart Attack'],
            y=['No Heart Attack', 'Heart Attack']
        )
        fig.update_layout(width=400, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (if available)
    if feature_importance is not None and feature_names is not None:
        st.subheader(f"📊 Feature Importance - {model_name}")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Top 10 Feature Importances - {model_name}",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation insights
    st.write(f"**🎯 {model_name} Performance Insights:**")
    st.write(f"• **Accuracy**: {metrics['accuracy']:.4f} - The model correctly predicts {metrics['accuracy']*100:.2f}% of all cases")
    st.write(f"• **Precision**: {metrics['precision']:.4f} - Of all positive predictions, {metrics['precision']*100:.2f}% are correct")
    st.write(f"• **Recall**: {metrics['recall']:.4f} - The model identifies {metrics['recall']*100:.2f}% of all actual positive cases")
    st.write(f"• **F1-Score**: {metrics['f1_score']:.4f} - Balanced measure of precision and recall")
    st.write(f"• **AUC**: {metrics['auc']:.4f} - Area under ROC curve (closer to 1.0 is better)")
    
    # Performance interpretation
    if metrics['auc'] >= 0.9:
        st.write("🌟 **Excellent performance** - Very strong predictive ability")
    elif metrics['auc'] >= 0.8:
        st.write("✅ **Good performance** - Strong predictive ability")
    elif metrics['auc'] >= 0.7:
        st.write("⚠️ **Fair performance** - Moderate predictive ability")
    else:
        st.write("❌ **Poor performance** - Limited predictive ability")
    
    st.markdown('</div>', unsafe_allow_html=True)


def train_random_forest(df):
    """Train and analyze Random Forest model"""
    st.markdown('<div class="sub-header">🌲 Random Forest Analysis</div>', unsafe_allow_html=True)
    
    # Prepare data
    data_tuple = prepare_ml_data(df)
    if data_tuple[0] is None:
        return
    
    X, y, X_train, X_test, y_train, y_test = data_tuple
    
    # Train model
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    execution_time = (time.time() - start_time) / 60  # Convert to minutes
    
    # Calculate metrics
    metrics = calculate_model_metrics(y_test, y_pred, y_pred_proba)
    
    # Display results with feature importance
    display_model_results("Random Forest", metrics, execution_time, y_test, y_pred, 
                         X.columns.tolist(), model.feature_importances_)

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
        title_x=0.5, 
        title_xanchor="center"
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
        height=300 * rows,
        title_text="Feature Distribution Analysis",
        showlegend=False,
        title_x=0.5,
        title_xanchor="center"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    summary_stats = df[selected_features].describe()
    st.dataframe(summary_stats.round(3))
    
    # Distribution insights
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("**🎯 Distribution Insights:**")
    for feature in selected_features:
        skewness = df[feature].skew()
        if abs(skewness) > 1:
            st.write(f"• {feature}: Highly skewed distribution (skewness: {skewness:.2f})")
        elif abs(skewness) > 0.5:
            st.write(f"• {feature}: Moderately skewed distribution (skewness: {skewness:.2f})")
        else:
            st.write(f"• {feature}: Normal distribution (skewness: {skewness:.2f})")
    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_visualizations(df):
    """Create advanced visualizations for deeper insights"""
    st.markdown('<div class="sub-header">🎨 Advanced Visualizations</div>', unsafe_allow_html=True)
    
    # Heart Attack Analysis by Demographics
    if 'HadHeartAttack' in df.columns:
        st.subheader("💔 Heart Attack Analysis by Demographics")
        
        demographic_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                demographic_cols.append(col)
        
        if demographic_cols:
            selected_demo = st.selectbox(
                "Select demographic variable:",
                demographic_cols,
                key="demo_selector"
            )
            
            if selected_demo:
                # Create crosstab
                crosstab = pd.crosstab(df[selected_demo], df['HadHeartAttack'], normalize='index') * 100
                
                fig = px.bar(
                    crosstab,
                    title=f"Heart Attack Rate by {selected_demo}",
                    labels={'value': 'Percentage (%)', 'index': selected_demo},
                    color_discrete_map={0: 'lightblue', 1: 'red'}
                )
                
                fig.update_layout(
                    xaxis_title=selected_demo,
                    yaxis_title="Heart Attack Rate (%)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Scatter Plot Matrix
    st.subheader("🔍 Scatter Plot Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 3:
        selected_scatter_vars = st.multiselect(
            "Select variables for scatter plot matrix:",
            numeric_cols,
            default=numeric_cols[:3],
            key="scatter_selector"
        )
        
        if len(selected_scatter_vars) >= 2:
            scatter_df = df[selected_scatter_vars].dropna()
            
            fig = px.scatter_matrix(
                scatter_df,
                title="Feature Relationships - Scatter Plot Matrix",
                height=600
            )
            
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)


def create_feature_importance_analysis(df):
    """Create comprehensive feature importance analysis"""
    st.markdown('<div class="sub-header">🎯 Feature Importance Analysis</div>', unsafe_allow_html=True)
    
    # Prepare data
    data_tuple = prepare_ml_data(df)
    if data_tuple[0] is None:
        return
    
    X, y, X_train, X_test, y_train, y_test = data_tuple
    
    # Train Random Forest for feature importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Top 15 features
    top_features = feature_importance.head(15)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 15 Feature Importance (Random Forest)",
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance insights
    st.write("**🎯 Feature Importance Insights:**")
    st.write(f"• **Most Important Feature:** {top_features.iloc[0]['Feature']} (Importance: {top_features.iloc[0]['Importance']:.4f})")
    st.write(f"• **Top 3 Features:** {', '.join(top_features.head(3)['Feature'].tolist())}")
    st.write(f"• **Cumulative Importance (Top 5):** {top_features.head(5)['Importance'].sum():.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">❤️ Heart Disease Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Team Members:** Arief bin Abdul Latib, Khor Cojean, Lim Yong Xiang, Ooi Yong Hang")

    # Sidebar
    st.sidebar.title("🔧 Dashboard Controls")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available. Please check the data file.")
        return
    
    # Data preprocessing
    df_processed, label_encoders = data_preprocessing(df)
    
    # Sidebar options
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        [
            "📊 Data Overview",
            "📚 Quick Reference Guide",
            "🔥 Correlation Analysis",
            "📈 Distribution Analysis",
            "🎨 Advanced Visualizations",
            "🌲 Random Forest",
            "🎯 Feature Importance"
        ]
    )
    
    # Display selected analysis
    if analysis_type == "📊 Data Overview":
        st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        # Overview metrics
        create_overview_metrics(df)
        
        # Data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Dataset Information")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"**Numeric Columns:** {len(df.select_dtypes(include=[np.number]).columns)}")
            st.write(f"**Categorical Columns:** {len(df.select_dtypes(include=['object']).columns)}")
        
        with col2:
            st.subheader("🧹 Data Quality")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                st.write("**Missing Values by Column:**")
                for col, missing_count in missing_data.items():
                    percentage = (missing_count / len(df)) * 100
                    st.write(f"• {col}: {missing_count} ({percentage:.1f}%)")
            else:
                st.success("✅ No missing values found!")
            
            # Duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.write(f"**Duplicate Rows:** {duplicates}")
            else:
                st.success("✅ No duplicate rows found!")
        
        # Sample data
        st.subheader("📝 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
        else:
            st.warning("No numeric columns found for statistical summary.")
    
    elif analysis_type == "📚 Quick Reference Guide":
        create_quick_reference_guide(df, df_processed, label_encoders)
    
    elif analysis_type == "🔥 Correlation Analysis":
        create_interactive_correlation_heatmap(df_processed)
    
    elif analysis_type == "📈 Distribution Analysis":
        create_distribution_analysis(df_processed)
    
    elif analysis_type == "🎨 Advanced Visualizations":
        create_advanced_visualizations(df_processed)
    
    elif analysis_type == "🌲 Random Forest":
        train_random_forest(df_processed)
    
    elif analysis_type == "🎯 Feature Importance":
        create_feature_importance_analysis(df_processed)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Use the sidebar to navigate between different analyses. "
        "Each analysis provides insights into the heart disease dataset."
    )

# Run the application
if __name__ == "__main__":
    main()
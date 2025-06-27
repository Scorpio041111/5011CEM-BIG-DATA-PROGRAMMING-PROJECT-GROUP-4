import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Clear matplotlib memory after each plot
def clear_matplotlib():
    plt.clf()
    plt.cla()
    plt.close()

# Load Excel file safely with error feedback
@st.cache_data
def load_data():
    excel_path = r"C:\Users\Xiang\20250627_Big_Data\Heart_2022_With_Nans.xlsx"
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        return df
    except FileNotFoundError:
        st.error(f"File not found: `{excel_path}`")
        return pd.DataFrame()
    except ImportError:
        st.error("`openpyxl` is not installed. Please install it using `pip install openpyxl`.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f" Unexpected error loading Excel file:\n\n{e}")
        return pd.DataFrame()

# Mean imputation for numeric fields
def impute_missing(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

# Heatmap of correlations
def plot_heatmap(df):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    clear_matplotlib()

# Histograms of features
def plot_histograms(df, columns):
    st.subheader("Feature Histograms")
    for col in columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
        clear_matplotlib()

# Train Random Forest model
def train_rf_model(df, target_col):
    df = df.select_dtypes(include=[np.number])
    if target_col not in df.columns:
        return None, None, None, None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    feature_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return acc, cm, report, feature_importance

# Main Streamlit app
def main():
    st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
    st.title("Heart Disease Full Analysis Dashboard")
    st.markdown("Group Members: Arief bin Abdul Latib, Khor Cojean, Lim Yong Xiang, Ooi Yong Hang")

    # Load and check data
    df_raw = load_data()
    if df_raw.empty:
        st.stop()

    df_clean = impute_missing(df_raw)

    # Raw dataset
    st.header("Raw Dataset (with Missing Values)")
    st.dataframe(df_raw)

    # Stats before cleaning
    st.header("Summary Statistics (Before Imputation)")
    st.write(df_raw.describe())

    # Cleaned dataset
    st.header(" Dataset After Mean Imputation")
    st.dataframe(df_clean)

    st.header(" Summary Statistics (After Imputation)")
    st.write(df_clean.describe())

    # Correlation heatmap
    plot_heatmap(df_clean)

    # Histograms of top 5 numeric features
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    plot_histograms(df_clean, numeric_cols[:5])

    # Machine Learning
    st.header("Random Forest Classifier Results")
    target = "HeartDisease"
    if target in df_clean.columns:
        acc, cm, report, importance = train_rf_model(df_clean, target)

        if acc is not None:
            st.success(f" Model Accuracy: {acc:.2f}")
            st.markdown("**Classification Report**")
            st.text(report)

            st.markdown("** Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            clear_matplotlib()

            st.markdown("** Feature Importance**")
            st.bar_chart(importance)
        else:
            st.warning(" Model training skipped. Target column not found in numeric data.")
    else:
        st.error(f" Target column `{target}` not found in the dataset.")

# Launch app
if __name__ == "__main__":
    main()

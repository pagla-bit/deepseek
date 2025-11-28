import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📊 Data Analysis Dashboard</h1>', unsafe_allow_html=True)
st.write("Upload your data file and explore it with interactive visualizations!")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", 
    ["Data Upload", "Data Overview", "Visualizations", "Statistical Analysis"])

# Sample data generation
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'sales': np.random.normal(1000, 200, 100).cumsum() + 1000,
        'customers': np.random.poisson(50, 100),
        'temperature': np.random.normal(25, 5, 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }
    return pd.DataFrame(data)

# File upload function
def handle_file_upload():
    st.markdown('<h2 class="section-header">📁 Upload Your Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        st.info("👆 Upload a file above or use our sample data to get started!")
        if st.button("Use Sample Data"):
            df = generate_sample_data()
            st.success("Sample data loaded successfully!")
            return df
    return None

# Data overview function
def show_data_overview(df):
    st.markdown('<h2 class="section-header">📈 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Basic statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# Visualization function
def show_visualizations(df):
    st.markdown('<h2 class="section-header">📊 Visualizations</h2>', unsafe_allow_html=True)
    
    # Select columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for visualization.")
        return
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Choose visualization type",
        ["Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Bar Chart", "Correlation Heatmap"]
    )
    
    col1, col2 = st.columns(2)
    
    if viz_type == "Scatter Plot":
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        
        if x_axis and y_axis:
            color_by = st.selectbox("Color by", [None] + categorical_cols)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                           title=f"Scatter Plot: {x_axis} vs {y_axis}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        with col1:
            x_axis = st.selectbox("X-axis", df.columns, key="line_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, key="line_y")
        
        if x_axis and y_axis:
            fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} over {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        column = st.selectbox("Select column", numeric_cols, key="hist_col")
        if column:
            bins = st.slider("Number of bins", 5, 100, 30)
            fig = px.histogram(df, x=column, nbins=bins, title=f"Histogram of {column}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        column = st.selectbox("Select column", numeric_cols, key="box_col")
        if column:
            fig = px.box(df, y=column, title=f"Box Plot of {column}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart":
        if categorical_cols:
            column = st.selectbox("Select categorical column", categorical_cols, key="bar_cat")
            if column:
                value_counts = df[column].value_counts().reset_index()
                value_counts.columns = [column, 'count']
                fig = px.bar(value_counts, x=column, y='count', 
                           title=f"Bar Chart of {column}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu_r',
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

# Statistical analysis function
def show_statistical_analysis(df):
    st.markdown('<h2 class="section-header">📐 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for statistical analysis.")
        return
    
    # Column selection for analysis
    selected_columns = st.multiselect(
        "Select columns for detailed analysis",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    
    if selected_columns:
        st.subheader("Detailed Statistics")
        detailed_stats = df[selected_columns].describe().T
        detailed_stats['variance'] = df[selected_columns].var()
        detailed_stats['skewness'] = df[selected_columns].skew()
        detailed_stats['kurtosis'] = df[selected_columns].kurtosis()
        st.dataframe(detailed_stats, use_container_width=True)
        
        # Correlation matrix
        if len(selected_columns) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[selected_columns].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), 
                        use_container_width=True)

# Main app logic
def main():
    df = None
    
    if app_mode == "Data Upload":
        df = handle_file_upload()
        if df is not None:
            st.session_state['df'] = df
    else:
        if 'df' not in st.session_state:
            st.warning("Please upload data first in the 'Data Upload' section.")
            return
        df = st.session_state['df']
    
    if df is not None:
        if app_mode == "Data Overview":
            show_data_overview(df)
        elif app_mode == "Visualizations":
            show_visualizations(df)
        elif app_mode == "Statistical Analysis":
            show_statistical_analysis(df)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Online Shopper Intention Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #cce5ff;
        color: #004085;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/shopping-cart--v2.png", width=100)
st.sidebar.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Analysis", "📈 Visualizations", "🔮 Prediction", "❓ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: center;'>Settings</h3>", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('online_shoppers_intention_edt.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function for data preprocessing
def preprocess_data(df):
    processed_df = df.copy()
    
    # Handle missing values
    numeric_columns = processed_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    
    # For numeric columns, fill NaN with median
    for col in numeric_columns:
        processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    # For categorical columns, fill NaN with mode
    for col in categorical_columns:
        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Drop duplicates
    processed_df.drop_duplicates(inplace=True)
    
    # Scale numerical features
    cols_to_scale = ['Administrative_Duration', 'Informational_Duration', 
                     'ProductRelated_Duration', 'BounceRates', 'ExitRates', 
                     'PageValues', 'SpecialDay']
    scaler = MinMaxScaler()
    processed_df[cols_to_scale] = scaler.fit_transform(processed_df[cols_to_scale])
    
    return processed_df

# Function to train model
def train_model(X, y):
    # Check for any remaining NaN values
    if X.isna().any().any():
        st.error("Data contains missing values after preprocessing. Please check the data.")
        return None, None, None, None, None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with st.spinner('Training the model...'):
        progress_bar = st.progress(0)
        
        try:
            # Apply SMOTE for handling class imbalance
            smote = SMOTE(sampling_strategy='minority', random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            # Initialize and train ExtraTreesClassifier
            et = ExtraTreesClassifier(n_estimators=100, random_state=42)
            et.fit(X_train_sm, y_train_sm)
            progress_bar.progress(50)
            
            # Evaluate model
            y_pred = et.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            progress_bar.progress(100)
            
            return et, X_test, y_test, y_pred, accuracy, report, cm
            
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            return None, None, None, None, None, None, None

# Home page content
if page == "🏠 Home":
    st.markdown("<h1 class='main-header'>Online Shopper Intention Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class='highlight'>
            <h2 class='sub-header'>Project Overview</h2>
            <p>This dashboard provides comprehensive tools for analyzing online shopping behavior and predicting purchase intentions. 
            Marketing teams can use these insights to optimize strategies for customer acquisition, retention, and revenue growth.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h3 class='section-header'>Problem Statement</h3>
            <p>Marketing teams constantly strive to optimize their promotions, pricing, personalization, and campaigns to increase 
            customer acquisition, retention, and revenue. However, identifying the most effective strategies can be challenging.</p>
            <p>This project aims to develop a machine learning model that can predict whether a customer visiting an online shopping 
            website will make a purchase or not. This prediction can help marketing teams in optimizing their strategies to increase 
            the likelihood of purchase and ultimately, revenue.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/color/240/000000/online-store.png", width=200)
    
    st.markdown("""
    <h2 class='sub-header'>Dashboard Features</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; border-radius: 5px; border: 1px solid #e0e0e0;'>
            <img src='https://img.icons8.com/color/96/000000/data-configuration.png' width='50'>
            <h3>Data Analysis</h3>
            <p>Explore and understand the online shopper dataset with descriptive statistics and dataset information.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; border-radius: 5px; border: 1px solid #e0e0e0;'>
            <img src='https://img.icons8.com/color/96/000000/combo-chart--v1.png' width='50'>
            <h3>Visualizations</h3>
            <p>Interactive visualizations to identify patterns and insights in customer behavior.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; border-radius: 5px; border: 1px solid #e0e0e0;'>
            <img src='https://img.icons8.com/color/96/000000/crystal-ball.png' width='50'>
            <h3>Prediction</h3>
            <p>Machine learning model to predict purchase intention based on various features.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; border-radius: 5px; border: 1px solid #e0e0e0;'>
            <img src='https://img.icons8.com/color/96/000000/info--v1.png' width='50'>
            <h3>About</h3>
            <p>Information about the project, data sources, and methodology.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Getting Started</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>Use the sidebar on the left to navigate between different sections of the dashboard. Here's a quick guide:</p>
    <ul>
        <li><strong>Data Analysis</strong>: Explore the dataset structure, summary statistics, and basic info</li>
        <li><strong>Visualizations</strong>: Explore interactive charts showing key insights and patterns</li>
        <li><strong>Prediction</strong>: Use the trained model to predict customer purchase intentions</li>
        <li><strong>About</strong>: Learn more about this project</li>
    </ul>
    """, unsafe_allow_html=True)

# Data Analysis page
elif page == "📊 Data Analysis":
    st.markdown("<h1 class='main-header'>Data Analysis</h1>", unsafe_allow_html=True)
    
    # Loading data with progress bar
    with st.spinner('Loading data...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        df = load_data()
    
    if df is not None:
        st.success("Data loaded successfully!")
        
        # Dataset overview
        st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p class='section-header'>Dataset Shape</p>", unsafe_allow_html=True)
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            st.markdown("<p class='section-header'>Data Types</p>", unsafe_allow_html=True)
            st.write(df.dtypes)
        
        with col2:
            st.markdown("<p class='section-header'>Missing Values</p>", unsafe_allow_html=True)
            missing_values = df.isnull().sum()
            st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")
            
            st.markdown("<p class='section-header'>Duplicate Rows</p>", unsafe_allow_html=True)
            st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
        
        # Data preview
        st.markdown("<h2 class='sub-header'>Data Preview</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 Sample Data", "📊 Summary Statistics", "🔍 Value Counts"])
        
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab3:
            col = st.selectbox("Select column for value counts:", df.columns)
            st.dataframe(df[col].value_counts(), use_container_width=True)
        
        # Class distribution
        st.markdown("<h2 class='sub-header'>Target Class Distribution</h2>", unsafe_allow_html=True)
        
        fig = px.pie(
            values=df['Revenue'].value_counts().values,
            names=['No Purchase (0)', 'Purchase (1)'],
            title="Revenue Class Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data preprocessing
        if st.checkbox("Show preprocessed data"):
            processed_df = preprocess_data(df)
            st.dataframe(processed_df.head(10), use_container_width=True)

# Visualizations page
elif page == "📈 Visualizations":
    st.markdown("<h1 class='main-header'>Data Visualizations</h1>", unsafe_allow_html=True)
    
    # Loading data with progress bar
    with st.spinner('Loading data...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        df = load_data()
        if df is not None:
            processed_df = preprocess_data(df)
    
    if df is not None:
        st.success("Data loaded successfully!")
        
        # Set sidebar options for visualizations
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h3 style='text-align: center;'>Visualization Options</h3>", unsafe_allow_html=True)
        viz_type = st.sidebar.selectbox(
            "Select Visualization Type",
            ["Feature Distributions", "Correlation Analysis", "Purchase Behavior", "Traffic Analysis"]
        )
        
        if viz_type == "Feature Distributions":
            st.markdown("<h2 class='sub-header'>Feature Distributions</h2>", unsafe_allow_html=True)
            
            # Select a feature to visualize
            numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            feature_to_visualize = st.selectbox("Select a feature to visualize:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution by feature
                fig = px.histogram(
                    processed_df, 
                    x=feature_to_visualize,
                    color="Revenue",
                    labels={"Revenue": "Purchase Made"},
                    title=f"Distribution of {feature_to_visualize}",
                    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
                    opacity=0.7
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    processed_df,
                    x="Revenue",
                    y=feature_to_visualize,
                    color="Revenue",
                    labels={"Revenue": "Purchase Made"},
                    title=f"Box Plot of {feature_to_visualize} by Purchase",
                    color_discrete_map={0: '#636EFA', 1: '#EF553B'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlation Analysis":
            st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
            
            # Calculate correlation matrix
            corr_matrix = processed_df.corr()
            
            # Heatmap
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations with Revenue
            st.markdown("<h3 class='section-header'>Top Correlations with Revenue</h3>", unsafe_allow_html=True)
            
            revenue_corr = corr_matrix['Revenue'].sort_values(ascending=False)
            fig = px.bar(
                x=revenue_corr.index,
                y=revenue_corr.values,
                labels={'x': 'Feature', 'y': 'Correlation with Revenue'},
                title="Feature Correlation with Revenue",
                color=revenue_corr.values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Purchase Behavior":
            st.markdown("<h2 class='sub-header'>Purchase Behavior Analysis</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Purchase by Visitor Type
                visitor_purchase = pd.crosstab(df['VisitorType'], df['Revenue'])
                visitor_purchase_pct = visitor_purchase.div(visitor_purchase.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    visitor_purchase_pct,
                    labels={'value': 'Percentage (%)', 'Revenue': 'Purchase'},
                    title="Purchase Rate by Visitor Type",
                    barmode='group',
                    color_discrete_map={0: '#636EFA', 1: '#EF553B'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Purchase by Weekend
                weekend_purchase = pd.crosstab(df['Weekend'], df['Revenue'])
                weekend_purchase_pct = weekend_purchase.div(weekend_purchase.sum(axis=1), axis=0) * 100
                
                fig = px.bar(
                    weekend_purchase_pct,
                    labels={'value': 'Percentage (%)', 'Revenue': 'Purchase'},
                    title="Purchase Rate by Weekend vs Weekday",
                    barmode='group',
                    color_discrete_map={0: '#636EFA', 1: '#EF553B'}
                )
                fig.update_xaxes(ticktext=["Weekday", "Weekend"], tickvals=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            # Purchase by Month
            month_purchase = pd.crosstab(df['Month'], df['Revenue'])
            month_purchase_pct = month_purchase.div(month_purchase.sum(axis=1), axis=0) * 100
            
            month_names = {2: 'Feb', 3: 'Mar', 5: 'May', 6: 'Jun', 7: 'Jul', 
                          8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            
            fig = px.line(
                x=month_purchase_pct.index.map(lambda x: month_names.get(x, x)),
                y=month_purchase_pct[1],
                markers=True,
                labels={'x': 'Month', 'y': 'Purchase Rate (%)'},
                title="Purchase Rate by Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Traffic Analysis":
            st.markdown("<h2 class='sub-header'>Traffic Analysis</h2>", unsafe_allow_html=True)
            
            # Traffic Type analysis
            traffic_purchase = pd.crosstab(df['TrafficType'], df['Revenue'])
            traffic_purchase_pct = traffic_purchase.div(traffic_purchase.sum(axis=1), axis=0) * 100
            
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "bar"}, {"type": "pie"}]],
                               subplot_titles=("Purchase Rate by Traffic Type", "Traffic Type Distribution"))
            
            # Bar chart for purchase rate
            fig.add_trace(
                go.Bar(
                    x=traffic_purchase_pct.index,
                    y=traffic_purchase_pct[1],
                    name="Purchase Rate",
                    marker_color='#EF553B'
                ),
                row=1, col=1
            )
            
            # Pie chart for traffic distribution
            fig.add_trace(
                go.Pie(
                    labels=df['TrafficType'].value_counts().index,
                    values=df['TrafficType'].value_counts().values,
                    name="Traffic Distribution"
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=500, title_text="Traffic Type Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Page values analysis
            st.markdown("<h3 class='section-header'>Page Values and Bounce Rates Analysis</h3>", unsafe_allow_html=True)
            
            fig = px.scatter(
                df,
                x='PageValues',
                y='BounceRates',
                color='Revenue',
                size='ProductRelated_Duration',
                hover_data=['VisitorType', 'TrafficType'],
                labels={
                    'PageValues': 'Page Values',
                    'BounceRates': 'Bounce Rates',
                    'Revenue': 'Purchase Made',
                    'ProductRelated_Duration': 'Time on Product Pages'
                },
                title="Relationship Between Page Values and Bounce Rates",
                color_discrete_map={0: '#636EFA', 1: '#EF553B'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Prediction page
elif page == "🔮 Prediction":
    st.markdown("<h1 class='main-header'>Purchase Intention Prediction</h1>", unsafe_allow_html=True)
    
    # Loading data with progress bar
    with st.spinner('Loading data...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        df = load_data()
        if df is not None:
            processed_df = preprocess_data(df)
    
    if df is not None:
        st.success("Data loaded successfully!")
        
        # Model training option
        st.markdown("<h2 class='sub-header'>Model Training and Evaluation</h2>", unsafe_allow_html=True)
        
        if st.button("Train Model"):
            # Prepare features and target
            try:
                X = pd.get_dummies(processed_df.drop('Revenue', axis=1))
                y = processed_df['Revenue']
                
                # Check for missing values before training
                if X.isna().any().any():
                    st.warning("Data contains missing values. Attempting to handle them...")
                    # Additional cleaning if needed
                    X = X.fillna(X.mean())
                
                # Train model
                model_results = train_model(X, y)
                
                if model_results[0] is not None:
                    model, X_test, y_test, y_pred, accuracy, report, cm = model_results
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<p class='section-header'>Model Performance</p>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='success-box'>
                            <h3>Model Accuracy: {accuracy:.2%}</h3>
                            <p>The model correctly predicts customer purchase intention {accuracy:.2%} of the time.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Feature importance
                        importances = model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            feature_importance.head(10),
                            x='Importance',
                            y='Feature',
                            title='Top 10 Feature Importances',
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("<p class='section-header'>Confusion Matrix</p>", unsafe_allow_html=True)
                        
                        # Plot confusion matrix
                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=['No Purchase', 'Purchase'],
                            y=['No Purchase', 'Purchase'],
                            color_continuous_scale='blues',
                            title="Confusion Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Classification report
                        st.markdown("<p class='section-header'>Classification Report</p>", unsafe_allow_html=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                else:
                    st.error("Model training failed. Please check your data and try again.")
            except Exception as e:
                st.error(f"Error preparing data for training: {str(e)}")
        
        # Prediction form
        st.markdown("<h2 class='sub-header'>Predict Purchase Intention</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p>Use the form below to input customer attributes and predict their purchase intention.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            admin_duration = st.slider("Administrative Duration (sec)", 0.0, 1000.0, 50.0)
            info_duration = st.slider("Informational Duration (sec)", 0.0, 1000.0, 50.0)
            product_duration = st.slider("Product Related Duration (sec)", 0.0, 1000.0, 100.0)
            bounce_rate = st.slider("Bounce Rate", 0.0, 1.0, 0.2)
            exit_rate = st.slider("Exit Rate", 0.0, 1.0, 0.2)
            page_value = st.slider("Page Value", 0.0, 100.0, 10.0)
        
        with col2:
            special_day = st.slider("Special Day (0-1)", 0.0, 1.0, 0.0)
            month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            os = st.selectbox("Operating System", list(range(9)))
            browser = st.selectbox("Browser", list(range(13)))
            region = st.selectbox("Region", list(range(10)))
            traffic_type = st.selectbox("Traffic Type", list(range(1, 21)))
            visitor_type = st.selectbox("Visitor Type", ['New_Visitor', 'Returning_Visitor', 'Other'])
            weekend = st.checkbox("Weekend")
        
        if st.button("Predict"):
            # Create a DataFrame with the user inputs
            user_input = {
                'Administrative_Duration': [admin_duration],
                'Informational_Duration': [info_duration],
                'ProductRelated_Duration': [product_duration],
                'BounceRates': [bounce_rate],
                'ExitRates': [exit_rate],
                'PageValues': [page_value],
                'SpecialDay': [special_day],
                'Month': [month],
                'OperatingSystems': [os],
                'Browser': [browser],
                'Region': [region],
                'TrafficType': [traffic_type],
                'VisitorType': [visitor_type],
                'Weekend': [weekend]
            }
            
            user_df = pd.DataFrame(user_input)
            
            # Preprocess the input data
            # Convert Weekend to numeric
            user_df['Weekend'] = 1 if weekend else 0
            
            # Convert Month to numeric
            month_mapping = {'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 
                             'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            user_df['Month'] = user_df['Month'].map(month_mapping)
            
            # Scale numerical features (same as in training)
            cols_to_scale = ['Administrative_Duration', 'Informational_Duration', 
                             'ProductRelated_Duration', 'BounceRates', 'ExitRates', 
                             'PageValues', 'SpecialDay']
            scaler = MinMaxScaler()
            
            # Fit scaler on processed_df to use same scaling
            scaler.fit(processed_df[cols_to_scale])
            user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])
            
            # One-hot encode categorical variables
            user_df = pd.get_dummies(user_df, columns=['VisitorType'])
            
            # Ensure all columns from training are present
            X = pd.get_dummies(processed_df.drop('Revenue', axis=1))
            for col in X.columns:
                if col not in user_df.columns:
                    user_df[col] = 0
            
            # Reorder columns to match training data
            user_df = user_df[X.columns]
            
            # Train model if not already trained
            X = pd.get_dummies(processed_df.drop('Revenue', axis=1))
            y = processed_df['Revenue']
            model, _, _, _, _, _, _ = train_model(X, y)
            
            # Make prediction
            prediction = model.predict(user_df)
            prediction_proba = model.predict_proba(user_df)
            
            # Display prediction result
            st.markdown("""
            <h3 class='section-header'>Prediction Result</h3>
            """, unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div style='background-color: #d4edda; color: #155724; padding: 20px; border-radius: 5px; text-align: center;'>
                    <h2>Will Make a Purchase! 🛒</h2>
                    <p>Confidence: {prediction_proba[0][1]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; text-align: center;'>
                    <h2>Unlikely to Purchase 🛑</h2>
                    <p>Confidence: {prediction_proba[0][0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display recommendations based on prediction
            st.markdown("<h3 class='section-header'>Marketing Recommendations</h3>", unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.markdown("""
                <div class='info-box'>
                    <h4>Recommendations for Likely Purchasers:</h4>
                    <ul>
                        <li>Offer complementary products to increase average order value</li>
                        <li>Implement loyalty programs to encourage repeat purchases</li>
                        <li>Ask for reviews or referrals after purchase</li>
                        <li>Provide exclusive deals to maintain interest</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature-specific recommendations
                if visitor_type == 'New_Visitor':
                    st.markdown("""
                    <div class='success-box'>
                        <h4>New Visitor Strategy:</h4>
                        <ul>
                            <li>Offer first-time purchase discount codes</li>
                            <li>Highlight easy return policy to build trust</li>
                            <li>Showcase testimonials from satisfied customers</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif visitor_type == 'Returning_Visitor':
                    st.markdown("""
                    <div class='success-box'>
                        <h4>Returning Visitor Strategy:</h4>
                        <ul>
                            <li>Personalize recommendations based on previous browsing history</li>
                            <li>Offer loyalty points for this purchase</li>
                            <li>Highlight new products since their last visit</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    <h4>Recommendations to Convert Non-Purchasers:</h4>
                    <ul>
                        <li>Implement exit-intent popups with special offers</li>
                        <li>Send abandoned cart emails if applicable</li>
                        <li>Improve page values by enhancing product descriptions and images</li>
                        <li>Offer first-time purchase discounts</li>
                        <li>Simplify the checkout process to reduce friction</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature-specific recommendations for non-purchasers
                if bounce_rate > 0.5:
                    st.markdown("""
                    <div class='warning-box'>
                        <h4>High Bounce Rate Strategy:</h4>
                        <ul>
                            <li>Improve landing page relevance to traffic source</li>
                            <li>Enhance site loading speed</li>
                            <li>Add compelling call-to-action above the fold</li>
                            <li>Implement A/B testing on landing pages</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                if page_value < 10:
                    st.markdown("""
                    <div class='warning-box'>
                        <h4>Low Page Value Strategy:</h4>
                        <ul>
                            <li>Enhance product images with zoom features and multiple angles</li>
                            <li>Add more detailed product descriptions</li>
                            <li>Include customer reviews and ratings</li>
                            <li>Show related products that might better match customer needs</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Insights based on traffic type
            traffic_insights = {
                2: "Traffic Type 2 typically has high conversion rates. Consider highlighting limited-time offers.",
                3: "Traffic Type 3 often responds well to detailed product information. Enhance descriptions and specifications.",
                4: "Traffic Type 4 visitors often compare prices. Highlight value propositions and competitive pricing.",
                1: "Traffic Type 1 users typically need more engagement. Try interactive elements and quizzes.",
                5: "Traffic Type 5 has higher conversion with visual content. Prioritize high-quality images and videos.",
                6: "Traffic Type 6 users respond well to social proof. Highlight reviews and testimonials.",
                7: "Traffic Type 7 converts better with simplified checkout. Minimize steps to purchase.",
                8: "Traffic Type 8 values security. Emphasize trust badges and secure payment methods.",
                9: "Traffic Type 9 responds to urgency. Consider countdown timers for limited offers.",
                10: "Traffic Type 10 benefits from personalized recommendations. Implement product suggestion algorithms."
            }
            
            if traffic_type in traffic_insights:
                st.markdown(f"""
                <div class='highlight'>
                    <h4>Traffic Type {traffic_type} Insight:</h4>
                    <p>{traffic_insights[traffic_type]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Conversion probability visualization
            st.markdown("<h3 class='section-header'>Purchase Probability</h3>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba[0][1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Purchase Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2C3E50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#F8D7DA"},
                        {'range': [30, 70], 'color': "#FFF3CD"},
                        {'range': [70, 100], 'color': "#D4EDDA"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # What-if analysis section
            st.markdown("<h3 class='section-header'>What-If Analysis</h3>", unsafe_allow_html=True)
            
            if st.checkbox("Explore how changing features affects the prediction"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Select feature to modify
                    feature_to_modify = st.selectbox(
                        "Select feature to modify:",
                        ["PageValues", "ProductRelated_Duration", "BounceRates", "ExitRates"]
                    )
                    
                    # Get current value
                    current_value = user_df[feature_to_modify].values[0]
                    
                    # Slider for new value
                    if feature_to_modify in ["BounceRates", "ExitRates"]:
                        new_value = st.slider(f"New {feature_to_modify} value:", 0.0, 1.0, float(current_value))
                    else:
                        max_val = 1.0 if current_value <= 1.0 else current_value * 2
                        new_value = st.slider(f"New {feature_to_modify} value:", 0.0, max_val, float(current_value))
                
                with col2:
                    # Create a copy of the user dataframe
                    modified_df = user_df.copy()
                    modified_df[feature_to_modify] = new_value
                    
                    # Get new prediction
                    new_prediction = model.predict(modified_df)
                    new_prediction_proba = model.predict_proba(modified_df)
                    
                    st.markdown("<p class='section-header'>Impact on Prediction</p>", unsafe_allow_html=True)
                    
                    if new_prediction[0] == prediction[0]:
                        change_text = "No change in prediction outcome."
                        if new_prediction[0] == 1:
                            confidence_change = new_prediction_proba[0][1] - prediction_proba[0][1]
                            change_text += f" Confidence {'increased' if confidence_change > 0 else 'decreased'} by {abs(confidence_change):.2%}."
                        else:
                            confidence_change = new_prediction_proba[0][0] - prediction_proba[0][0]
                            change_text += f" Confidence {'increased' if confidence_change > 0 else 'decreased'} by {abs(confidence_change):.2%}."
                    else:
                        change_text = f"Prediction changed from {'Purchase' if prediction[0] == 1 else 'No Purchase'} to {'Purchase' if new_prediction[0] == 1 else 'No Purchase'}!"
                    
                    if new_prediction[0] == 1:
                        st.markdown(f"""
                        <div style='background-color: #d4edda; color: #155724; padding: 20px; border-radius: 5px;'>
                            <h4>New Prediction: Will Make a Purchase! 🛒</h4>
                            <p>Confidence: {new_prediction_proba[0][1]:.2%}</p>
                            <p>{change_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px;'>
                            <h4>New Prediction: Unlikely to Purchase 🛑</h4>
                            <p>Confidence: {new_prediction_proba[0][0]:.2%}</p>
                            <p>{change_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Export prediction
            st.markdown("<h3 class='section-header'>Export Prediction</h3>", unsafe_allow_html=True)
            
            if st.button("Export Prediction Results"):
                # Create prediction summary
                prediction_summary = pd.DataFrame({
                    'Feature': user_df.columns,
                    'Value': user_df.values[0]
                })
                
                prediction_summary = prediction_summary.append({
                    'Feature': 'Prediction',
                    'Value': 'Purchase' if prediction[0] == 1 else 'No Purchase'
                }, ignore_index=True)
                
                prediction_summary = prediction_summary.append({
                    'Feature': 'Confidence',
                    'Value': f"{prediction_proba[0][1]:.2%}" if prediction[0] == 1 else f"{prediction_proba[0][0]:.2%}"
                }, ignore_index=True)
                
                # Convert to CSV
                csv = prediction_summary.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download Prediction as CSV",
                    data=csv,
                    file_name="shopper_prediction.csv",
                    mime="text/csv"
                )
                
                st.success("Prediction report ready for download!")

# About page
elif page == "❓ About":
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
        <h2 class='sub-header'>Project Background</h2>
        <p>This dashboard was developed to help marketing teams analyze and predict online shopper behavior.
        The predictive model helps identify potential customers who are more likely to make a purchase,
        allowing for more targeted marketing strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3 class='section-header'>Dataset Information</h3>
        <p>The "Online Shoppers Purchasing Intention Dataset" from UCI Machine Learning Repository contains
        various features related to user behavior on an online shopping website, including:</p>
        <ul>
            <li>Administrative, informational, and product-related page visits and duration</li>
            <li>Bounce rates and exit rates</li>
            <li>Page values and special day proximity</li>
            <li>Visitor demographics and technical information</li>
            <li>Time-related information (month, weekend)</li>
        </ul>
        <p>The target variable "Revenue" indicates whether a purchase was made or not.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h3 class='section-header'>Methodology</h3>
        <p>The project follows these key steps:</p>
        <ol>
            <li><strong>Data Preprocessing</strong>: Cleaning, encoding categorical variables, handling missing values, and scaling</li>
            <li><strong>Exploratory Data Analysis</strong>: Statistical analysis and visualization of key patterns</li>
            <li><strong>Model Training</strong>: Using Extra Trees Classifier for prediction after applying SMOTE for class imbalance</li>
            <li><strong>Feature Importance Analysis</strong>: Identifying key factors that influence purchase decisions</li>
            <li><strong>Interactive Prediction</strong>: Allowing users to input customer attributes and get purchase predictions</li>
        </ol>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 class='section-header'>How to Use This Dashboard</h3>
    <p>This dashboard is designed to be intuitive and user-friendly:</p>
    <ul>
        <li>Use the sidebar to navigate between different sections</li>
        <li>Explore the data analysis section to understand the dataset characteristics</li>
        <li>View visualizations to identify patterns and insights</li>
        <li>Train the model and use the prediction tool to forecast purchase intentions</li>
        <li>Apply the recommendations to optimize marketing strategies based on predictions</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h3 class='section-header'>Key Insights from the Model</h3>
    <div class='info-box'>
        <ul>
            <li><strong>Page Value</strong> is one of the strongest predictors of purchase intent - optimizing product pages with clear value propositions can significantly increase conversion rates</li>
            <li><strong>Exit Rate</strong> and <strong>Bounce Rate</strong> are negatively correlated with purchases - improving user engagement and reducing early exits is crucial</li>
            <li><strong>Product Related Duration</strong> shows a positive correlation with purchases - encouraging users to spend more time browsing products can lead to higher conversions</li>
            <li>Seasonal patterns exist, with certain months showing higher conversion rates - tailoring marketing campaigns to these seasonal trends is recommended</li>
            <li>Different visitor types show distinct behaviors - personalized strategies for new versus returning visitors yield better results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information and resources
    st.markdown("""
    <h3 class='section-header'>Resources and References</h3>
    <ul>
        <li>Dataset Source: <a href="https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset">UCI Machine Learning Repository</a></li>
        <li>Built with: Streamlit, Scikit-learn, Pandas, Plotly</li>
        <li>For more information, check the <a href="https://github.com/yourusername/shopper-intention-dashboard">GitHub repository</a></li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Version info
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 30px;'>
        <p style='text-align: center; margin: 0;'><small>Version 1.0.0 | Last Updated: March 2025</small></p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>© 2025 Online Shopper Intention Prediction Dashboard</p>
</div>
""", unsafe_allow_html=True)

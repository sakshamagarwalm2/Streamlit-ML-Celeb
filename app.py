import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="🏠 Advanced House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data generation and model training
@st.cache_data
def generate_house_data(n_samples=1000):
    """Generate synthetic house data with multiple features"""
    np.random.seed(42)
    
    # Generate correlated features
    size = np.random.normal(2000, 600, n_samples)
    bedrooms = np.random.poisson(3, n_samples) + 1
    bathrooms = np.random.poisson(2, n_samples) + 1
    age = np.random.randint(1, 50, n_samples)
    garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1])
    
    # Create price with realistic relationships
    price = (size * 80 + 
             bedrooms * 15000 + 
             bathrooms * 10000 + 
             (50 - age) * 500 + 
             garage * 8000 + 
             np.random.normal(0, 15000, n_samples))
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    return pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'garage_spaces': garage,
        'price': price
    })

@st.cache_resource
def train_models():
    """Train multiple ML models and return the best one"""
    df = generate_house_data()
    
    # Prepare features and target
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage_spaces']]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results = {
            'model': model,
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'predictions': y_pred,
            'actual': y_test
        }
        model_results[name] = results
    
    return model_results, X_train, X_test, y_train, y_test

def create_prediction_plot(model_results, selected_model):
    """Create prediction vs actual plot"""
    results = model_results[selected_model]
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=results['actual'],
        y=results['predictions'],
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='rgba(31, 119, 180, 0.6)',
            line=dict(width=1, color='rgba(31, 119, 180, 1)')
        )
    ))
    
    # Add perfect prediction line
    min_val = min(results['actual'].min(), results['predictions'].min())
    max_val = max(results['actual'].max(), results['predictions'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{selected_model} - Predictions vs Actual',
        xaxis_title='Actual Price ($)',
        yaxis_title='Predicted Price ($)',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot for Random Forest"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_names,
                y=importances,
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance',
            template='plotly_white'
        )
        
        return fig
    return None

def create_data_distribution_plot(df):
    """Create distribution plots for all features"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Size (sq ft)', 'Bedrooms', 'Bathrooms', 'Age (years)', 'Garage Spaces', 'Price'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    features = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage_spaces', 'price']
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (feature, pos) in enumerate(zip(features, positions)):
        fig.add_trace(
            go.Histogram(x=df[feature], name=feature, showlegend=False),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        title='Data Distribution',
        template='plotly_white',
        height=600
    )
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏠 Advanced House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection and inputs
    st.sidebar.header("🔧 Model Configuration")
    
    # Train models
    with st.spinner("Training machine learning models..."):
        model_results, X_train, X_test, y_train, y_test = train_models()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_results.keys()),
        index=1  # Default to Random Forest
    )
    
    # Display model metrics
    st.sidebar.subheader("📊 Model Performance")
    results = model_results[selected_model]
    
    st.sidebar.metric("R² Score", f"{results['r2_score']:.3f}")
    st.sidebar.metric("MAE", f"${results['mae']:,.0f}")
    st.sidebar.metric("RMSE", f"${np.sqrt(results['mse']):,.0f}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏡 House Details")
        
        # Input fields
        size = st.number_input(
            'House Size (sq ft)',
            min_value=500,
            max_value=5000,
            value=2000,
            step=50
        )
        
        bedrooms = st.selectbox(
            'Number of Bedrooms',
            options=[1, 2, 3, 4, 5, 6],
            index=2
        )
        
        bathrooms = st.selectbox(
            'Number of Bathrooms',
            options=[1, 2, 3, 4, 5],
            index=1
        )
        
        age = st.slider(
            'House Age (years)',
            min_value=1,
            max_value=50,
            value=10
        )
        
        garage = st.selectbox(
            'Garage Spaces',
            options=[0, 1, 2, 3],
            index=2
        )
        
        # Prediction button
        if st.button('🔮 Predict Price', type='primary'):
            # Prepare input data
            input_data = np.array([[size, bedrooms, bathrooms, age, garage]])
            
            # Make prediction
            model = model_results[selected_model]['model']
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🏠 Estimated House Price</h3>
                <h2 style="color: #28a745;">${prediction:,.0f}</h2>
                <p>Based on {selected_model} model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Store prediction in session state for visualization
            st.session_state.prediction = prediction
            st.session_state.input_data = {
                'size': size,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'garage': garage
            }
    
    with col2:
        st.subheader("📈 Model Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Predictions", "Feature Importance", "Data Distribution"])
        
        with tab1:
            # Prediction plot
            fig = create_prediction_plot(model_results, selected_model)
            
            # Add user prediction if available
            if hasattr(st.session_state, 'prediction'):
                # For visualization, we'll use the average actual price as reference
                avg_actual = y_test.mean()
                fig.add_trace(go.Scatter(
                    x=[avg_actual],
                    y=[st.session_state.prediction],
                    mode='markers',
                    name='Your Prediction',
                    marker=dict(size=15, color='red', symbol='star')
                ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Feature importance (only for Random Forest)
            if selected_model == 'Random Forest':
                feature_names = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Garage']
                fig = create_feature_importance_plot(
                    model_results[selected_model]['model'],
                    feature_names
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance is only available for Random Forest model.")
        
        with tab3:
            # Data distribution
            df = generate_house_data()
            fig = create_data_distribution_plot(df)
            st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights section
    st.subheader("🔍 Model Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Training Samples",
            "1,000",
            help="Number of samples used to train the model"
        )
    
    with col2:
        st.metric(
            "Features Used",
            "5",
            help="Size, bedrooms, bathrooms, age, garage spaces"
        )
    
    with col3:
        st.metric(
            "Model Accuracy",
            f"{results['r2_score']:.1%}",
            help="Percentage of variance explained by the model"
        )
    
    # Data exploration section
    with st.expander("🔍 Explore Training Data"):
        df = generate_house_data()
        st.dataframe(df.describe())
        
        # Correlation matrix
        st.subheader("Feature Correlation")
        correlation_matrix = df.corr()
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit | Machine Learning Model Deployment</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
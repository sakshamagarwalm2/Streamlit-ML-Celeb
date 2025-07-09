# Streamlit-ML-Celeb

## Overview
Streamlit-ML-Celeb is a machine learning-powered web application built with Streamlit to predict house prices based on various features such as size, number of bedrooms, bathrooms, age, and garage spaces. The application supports two machine learning models—Linear Regression and Random Forest—and provides interactive visualizations to explore model performance, feature importance, and data distributions.

## Features
- **Interactive Interface**: Input house details and get real-time price predictions.
- **Model Selection**: Choose between Linear Regression and Random Forest models.
- **Visualizations**:
  - Prediction vs. Actual scatter plot with user prediction highlighting.
  - Feature importance plot (for Random Forest model).
  - Data distribution histograms for all features.
  - Feature correlation matrix.
- **Model Performance Metrics**: Displays R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
- **Data Exploration**: View summary statistics and correlations of the training data.

## Installation
To run this application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sakshamagarwalm2/Streamlit-ML-Celeb
   cd sakshamagarwalm2-streamlit-ml-celeb
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Requirements
The required Python packages are listed in `requirements.txt`:
- streamlit==1.29.0
- pandas==2.1.4
- numpy==1.26.2
- scikit-learn==1.3.2
- plotly==5.18.0

## Usage
1. Open the application in your browser (default: `http://localhost:8501`).
2. Use the sidebar to:
   - Select a machine learning model (Linear Regression or Random Forest).
   - Input house details (size, bedrooms, bathrooms, age, garage spaces).
3. Click the **Predict Price** button to see the estimated house price.
4. Explore visualizations and model insights in the main panel:
   - **Predictions Tab**: View predicted vs. actual prices.
   - **Feature Importance Tab**: Analyze feature contributions (Random Forest only).
   - **Data Distribution Tab**: See histograms of all features.
5. Expand the **Explore Training Data** section to view summary statistics and the feature correlation matrix.

## Deployment
The application is deployed on Streamlit Community Cloud. Access it at:  
[https://app-ml-celeb-hldzmsw8ss4fgarsvuynxg.streamlit.app/](https://app-ml-celeb-hldzmsw8ss4fgarsvuynxg.streamlit.app/)

## Project Structure
```
sakshamagarwalm2-streamlit-ml-celeb/
├── README.md          # Project documentation
├── app.py            # Main Streamlit application
└── requirements.txt   # Python dependencies
```

## Notes
- The application uses synthetic house data generated with realistic relationships between features and prices.
- Random Forest typically provides better predictions due to its ability to capture non-linear relationships.
- The application is optimized for performance with caching for data generation and model training.

## License
This project is licensed under the MIT License.
Deployid on 
![https://app-ml-celeb-hldzmsw8ss4fgarsvuynxg.streamlit.app/]

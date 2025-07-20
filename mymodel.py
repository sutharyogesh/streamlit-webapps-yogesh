import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Synthetic generation of data examples for training the model
def generate_house_data(n_samples=100):
    np.random.seed(42)
    # Generate size data with a slightly wider range to show more variability
    size = np.random.normal(1800, 700, n_samples) 
    # Ensure sizes are positive and within a reasonable range for houses
    size = np.maximum(500, size) 
    size = np.minimum(5000, size)
    
    # Introduce a slight non-linearity or noise for more realistic data
    price = size * 120 + np.random.normal(0, 25000, n_samples) + (size / 100)**2 * 50
    price = np.maximum(50000, price) # Ensure prices are not unrealistically low
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Function for instantiating and training linear regression model
@st.cache_data # Cache the model training to avoid re-training on every rerun
def train_model():
    df = generate_house_data()
    
    # Train-test data splitting
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model (for display purposes)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_test, y_test # Return test data for evaluation visualization

# Streamlit User Interface for Deployed Model
def main():
    st.set_page_config(page_title="House Pricing Predictor", layout="centered")

    st.title('🏠 Simple House Pricing Predictor')
    st.write('Introduce the house size to predict its sale price.')
    
    # Train model and get evaluation metrics
    model, mse, r2, X_test, y_test = train_model()
    
    # Model Performance Section (New Feature)
    with st.expander("Model Performance Metrics"):
        st.write(f"**Mean Squared Error (MSE):** ${mse:,.2f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")
        st.info("R-squared indicates the proportion of the variance in the dependent variable (price) that is predictable from the independent variable (size). Higher is better (closer to 1).")
        
        # Plotting actual vs predicted for test set
        test_predictions = model.predict(X_test)
        test_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': test_predictions, 'Size': X_test['size_sqft']})
        
        fig_test = px.scatter(test_df, x='Actual Price', y='Predicted Price', 
                              hover_data=['Size'],
                              title='Actual vs. Predicted Prices (Test Set)',
                              labels={'Actual Price': 'Actual Price ($)', 'Predicted Price': 'Predicted Price ($)'})
        fig_test.add_trace(px.line(x=[test_df['Actual Price'].min(), test_df['Actual Price'].max()], 
                                   y=[test_df['Actual Price'].min(), test_df['Actual Price'].max()],
                                   color_discrete_sequence=['red']).data[0])
        st.plotly_chart(fig_test)
        st.write("The red line represents perfect predictions. Points closer to this line indicate better model performance.")


    st.markdown("---") # Separator

    # User input
    st.subheader("Make a Prediction")
    size = st.number_input('House size (square feet)', 
                           min_value=500, 
                           max_value=5000, 
                           value=1500,
                           step=100) # Added step for easier input
    
    if st.button('Predict price', help="Click to get the estimated price for the given house size"):
        # Perform prediction
        prediction = model.predict([[size]])
        
        # Show result
        st.success(f'Estimated price: **${prediction[0]:,.2f}**')
        
        # Visualization
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', 
                         title='House Size vs. Price Relationship with Prediction',
                         labels={'size_sqft': 'House Size (sqft)', 'price': 'Price ($)'},
                         hover_data={'size_sqft': ':.0f', 'price': '$,.2f'}) # Improved hover info

        # Add the regression line
        fig.add_trace(px.line(x=df['size_sqft'], y=model.predict(df[['size_sqft']]), 
                              color_discrete_sequence=['blue'], 
                              labels={'y':'Regression Line'}).data[0])

        fig.add_scatter(x=[size], y=[prediction[0]], 
                        mode='markers+text', # Added text for clarity
                        marker=dict(size=18, color='red', symbol='star'), # Changed marker style
                        name='Your Prediction',
                        text=[f'Predicted: ${prediction[0]:,.0f}'],
                        textposition="top center")
        
        st.plotly_chart(fig)
        st.write("The scatter points show the synthetic data used for training. The blue line is the learned regression line, and the red star indicates your prediction.")

    st.markdown("---")
    st.caption("Developed by Yogesh Suthar, 2025. All rights reserved. | [GitHub](https://github.com/sutharyogesh) ")

if __name__ == '__main__':
    main()
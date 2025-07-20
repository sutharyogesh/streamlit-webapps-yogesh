
# Simple House Pricing Predictor

This is a Streamlit app that predicts house prices based on their size using a linear regression model trained on synthetic data.

## Features

- Enter house size (in square feet) to get a price prediction.
- Visualizes the relationship between house size and price.
- Uses scikit-learn for model training and Plotly for interactive charts.

## Requirements

See [requirements.txt](requirements.txt) for dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- plotly

## Usage

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the app:
    ```sh
    streamlit run mymodel.py
    ```

3. Open the provided local URL in your browser.

## Files

- [`mymodel.py`](mymodel.py): Main app and model code.
- [`requirements.txt`](requirements.txt): Python dependencies.

## License

MIT License.

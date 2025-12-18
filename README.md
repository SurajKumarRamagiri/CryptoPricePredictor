# CryptoSense AI

**CryptoSense AI** is a real-time cryptocurrency price prediction application powered by deep learning models (LSTM & GRU). It allows users to visualize market trends, forecast future prices for major cryptocurrencies, and compare different model architectures for accuracy.

🌐 **Live Demo:** [https://cryptosense-ai.streamlit.app](https://cryptosense-ai.streamlit.app)

## Key Features

*   **Real-Time Data**: Fetches live market data from Yahoo Finance / Binance.
*   **Multi-Asset Support**: Predict prices for Bitcoin (BTC), Ethereum (ETH), and Litecoin (LTC).
*   **Flexible Forecasting**: Choose from multiple forecast horizons (1 Hour, 4 Hours, 24 Hours).
*   **Advanced AI Models**:
    *   **LSTM** (Long Short-Term Memory)
    *   **GRU** (Gated Recurrent Unit)
    *   **Auto Mode**: Automatically selects the best-performing model.
*   **Technical Analysis Dashboard**:
    *   Integrated overlay of **Bollinger Bands** on price charts.
    *   Dedicated subplots for **RSI** (Relative Strength Index) with Overbought/Oversold annotations.
    *   **MACD** (Moving Average Convergence Divergence) with signal lines and histograms.
*   **Portfolio Simulator**: Estimate potential Profit/Loss (PnL) based on your current holdings and model predictions.
*   **Model Comparison**: Side-by-side comparison of LSTM and GRU performance with aligned visualizations, detailed KPIs, and model-wise prediction tables.
*   **Interactive Visualizations**: Dynamic, zoomable Plotly charts that unify historical data, technical indicators, and future forecasts into a single compact view.

## Installation & Usage

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package installer)

### Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd CryptoPricePredictor
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **Access the App**:
    The application will open automatically in your browser. If not, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## Project Structure

*   `streamlit_app.py`: The main entry point for the Streamlit web application. Handles the UI and user interaction.
*   `src/`: Contains the core source code modules:
    *   `data.py`: Handles fetching data from external APIs (Yahoo Finance / Binance).
    *   `preprocessing.py`: Feature engineering and data scaling.
    *   `modeling.py`: Definitions for LSTM/GRU configurations and prediction logic.
    *   `visualization.py`: Functions for generating Plotly charts and UI metrics.
    *   `utils.py`: Configuration and helper utilities.
*   `AnalysisData/`: Directory where trained Model files and Scalers are stored.
*   `requirements.txt`: List of Python dependencies.

## How It Works

1.  **Select a Cryptocurrency**: Choose a pair (e.g., BTC-USD) from the sidebar.
2.  **Choose a Horizon**: Select how far into the future you want to predict (e.g., 1 Hour).
3.  **Pick a Model**: Select which model architecture to use, or let "Auto" decide.
4.  **Predict**: Click "Predict" to see the forecast, confidence intervals, and key metrics.
5.  **Compare**: Click "Compare" to run both LSTM and GRU models simultaneously and analyze their differences.

---
*Powered by Streamlit & TensorFlow*
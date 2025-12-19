"""
CryptoSense AI - Crypto Currency Price Predictor
Main Streamlit UI application (UI layer only)
Modular architecture with separate data, modeling, and visualization layers.
"""
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

from src.data import DataFetcher
from src.preprocessing import FeatureProcessor
from src.modeling import PredictionEngine
from src.visualization import Visualizer
from src.utils import Config, DataLoader, TimelineBuilder
from src.views import View


# ============================
# UI: INPUT HANDLER
# ============================
class InputHandler:
    
    @staticmethod
    def render_sidebar():
        """Render sidebar UI and capture user selections."""
        st.sidebar.markdown(
            """
            <div style="
                margin-top:-35px; 
                padding-top:0px;
                padding-bottom:10px;
                text-align:center;
            ">
                <h1 style="
                    font-size: 42px; 
                    font-weight: 800; 
                    margin-bottom: -10px;
                    color: #00FFFF;
                    letter-spacing: 1px;
                ">
                CryptoSense AI
                </h1>
                <p style="
                    font-size: 13px; 
                    color: #888888; 
                    margin-top: 0px;
                ">
                    Real-Time Market Intelligence
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.caption("Real-Time Market Intelligence AI-powered forecasts")

        theme_mode = "Dark"

        if theme_mode == "Dark":
            secondary_bg = "#151920"
            accent = "#00FFFF"
            button_bg = "#00FFFF"
            button_fg = "#000000"
        else:
            secondary_bg = "#ffffff"
            accent = "#0066cc"
            button_bg = "#0066cc"
            button_fg = "#ffffff"

        css_sidebar = f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: {secondary_bg};
            padding: 16px 12px 24px 12px;
            border-right: 1px solid rgba(0,0,0,0.04);
        }}
        @media (min-width: 768px) {{
            [data-testid="stSidebar"][aria-expanded="true"] {{
                min-width: 400px;
                max-width: 400px;
            }}
        }}
        @media (max-width: 768px) {{
            [data-testid="stSidebar"][aria-expanded="true"] {{
                min-width: 100%;
                max-width: 100%;
            }}
        }}
        .sidebar-title {{ font-size:20px; font-weight:700; color:{accent}; margin-bottom:6px; }}
        .sidebar-caption {{ font-size:12px; color:rgba(0,0,0,0.45); margin-top:-8px; margin-bottom:12px; }}
        div.stButton > button:first-child {{
        background-color: {button_bg};
        color: {button_fg};
        border-radius: 8px;
        padding: 10px 12px;
        font-weight:600;
        width:100%;
        }}
        div.stButton > button:first-child:hover {{
        opacity:0.92;
        transform: translateY(-1px);
        }}
        </style>
        """
        st.sidebar.markdown(css_sidebar, unsafe_allow_html=True)

        st.sidebar.markdown("### ⚙️ Trading Configuration")
        
        data_source = st.sidebar.radio(
            "Data Source",
            ("Binance", "Yahoo Finance"),
            index=0,
            horizontal=True
        )

        col = st.sidebar.columns([220, 1])[0]

        with col:
            pair_label = st.selectbox(
                "Select Cryptocurrency",
                (
                    "Bitcoin (BTC)", 
                    "Ethereum (ETH)", 
                    "Litecoin (LTC)"
                ),
                index=0
            )

        pair_map = {
            "Bitcoin (BTC)": "BTC-USD", 
            "Ethereum (ETH)": "ETH-USD", 
            "Litecoin (LTC)": "LTC-USD"
        }
        pair = pair_map.get(pair_label, "BTC-USD")

        horizon_label = st.sidebar.radio(
            "Forecast Period",
            ("1 Hour", "4 Hours", "24 Hours"),
            index=0
        )
        horizon_map = {"1 Hour": "1h", "4 Hours": "4h", "24 Hours": "24h"}
        horizon = horizon_map.get(horizon_label, "1h")

        st.sidebar.markdown("### 🧠 Model Selection")
        col_model = st.sidebar.columns([220, 1])[0]
        with col_model:
            model_choice = st.selectbox(
                "Pick model architecture (or Auto for best available)",
                ("Auto", "LSTM", "GRU"),
                index=0
            )
        st.sidebar.markdown("### 📅 Prediction Period")
        col_pred = st.sidebar.columns([220, 1])[0]
        
        # Dynamic Options based on Horizon
        if horizon == "1h":
            options = ("4 Hours", "8 Hours")
        elif horizon == "4h":
            options = ("12 Hours", "24 Hours")
        else: # 24h
            options = ("3 Days", "7 Days")
            
        with col_pred:
            prediction_period_label = st.selectbox(
                "Predict ahead for",
                options,
                index=0,
                key="prediction_period_select"
            )
        
        val = int(prediction_period_label.split()[0])
        if "Day" in prediction_period_label:
            prediction_period_hours = val * 24
        else:
            prediction_period_hours = val

        st.sidebar.markdown("### 💼 Portfolio Simulator")
        col_holdings = st.sidebar.columns([220, 1])[0]
        with col_holdings:
            with st.expander("Configure Holdings", expanded=False):
                holdings = st.number_input(
                    f"Amount of {pair.split('-')[0]} you hold:",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    format="%.4f"
                )

        st.sidebar.markdown("### 📈 Chart Options")
        show_indicators = st.sidebar.checkbox("Show Technical Indicators", value=False)

        col1, col2 = st.sidebar.columns([1, 1])

        with col1:
            run = st.button("🔮 Predict", key="predict_btn")

        with col2:
            compare = st.button("⚖️ Compare", key="compare_btn")

        st.sidebar.markdown("""
        <style>
        button[kind="primary"][data-testid="baseButton-predict_btn"] {
            background-color: #00FFFF !important;
            color: #000000 !important;
            border-radius: 8px !important;
            height: 45px !important;
        }

        button[kind="primary"][data-testid="baseButton-compare_btn"] {
            background-color: #8A2BE2 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            height: 45px !important;
        }

        button[kind="primary"][data-testid^="baseButton-"]:hover {
            opacity: 0.92 !important;
            transform: translateY(-1px);
        }
        </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔗 Powered by:** Binance API & Yahoo Finance")
        st.sidebar.markdown("**⏱ Timezone:** IST ")
        st.sidebar.caption("Built with Streamlit • Model: LSTM / GRU")

        if st.sidebar.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success("Cache Cleared!")

        return {
            "pair": pair,
            "horizon": horizon,
            "theme_mode": theme_mode,
            "model_choice": model_choice,
            "prediction_period_hours": prediction_period_hours,
            "run": run,
            "compare": compare,
            "source": data_source,
            "holdings": holdings,
            "show_indicators": show_indicators
        }

    
    @staticmethod
    def validate_config(pair, horizon):
        """Validate user configuration."""
        cfg = Config.MODEL_MAP.get((pair, horizon))
        if cfg is None:
            # Check fallback
            if pair not in Config.FALLBACK_CHAIN:
                # st.error("No model configured for this pair/horizon. Update Config.MODEL_MAP in the app.")
                return False
        return True


# ============================
# MAIN APP
# ============================
def main():
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="CryptoSense AI", page_icon="CPP_logo.png", layout="wide")
    
    # 1. Render Header
    View.render_header()

    # 2. Render Sidebar & Inputs
    user_input = InputHandler.render_sidebar()

    st.markdown(
        f"<div style='color:#9AA2AA; margin-bottom:0px;'>"
        f"<strong>Pair:</strong> {user_input['pair']} &nbsp; • &nbsp; <strong>Horizon:</strong> {user_input['horizon']}"
        f"</div>",
        unsafe_allow_html=True
    )

    # 3. Handle Main Logic Flow
    if user_input['run']:
        View.close_sidebar()
        if not InputHandler.validate_config(user_input['pair'], user_input['horizon']):
             st.error(f"⚠️ No AI models available for {user_input['pair']} ({user_input['horizon']}). Prediction disabled.")
             st.info("💡 You can currently only view historical data for this asset.")
             return
        View.render_prediction_flow(user_input)
    
    elif user_input['compare']:
        View.close_sidebar()
        if not InputHandler.validate_config(user_input['pair'], user_input['horizon']):
             st.error(f"⚠️ No AI models available for {user_input['pair']}. Comparison disabled.")
             return
        View.render_comparison_flow(user_input)
        
    else:
        # Default Idle View
        View.render_market_overview(user_input)

    # 4. Render Footer
    View.render_footer()


if __name__ == "__main__":
    main()

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
        st.sidebar.caption("Real-Time Market Intelligence — AI-powered forecasts")

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

        col = st.sidebar.columns([220, 1])[0]

        with col:
            pair_label = st.selectbox(
                "Select Cryptocurrency",
                ("Bitcoin (BTC)", "Ethereum (ETH)", "Litecoin (LTC)"),
                index=0
            )

        pair_map = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Litecoin (LTC)": "LTC-USD"}
        pair = pair_map.get(pair_label, "BTC-USD")

        horizon_label = st.sidebar.radio(
            "Forecast Period",
            ("1 Hour", "4 Hours", "24 Hours"),
            index=0
        )
        horizon_map = {"1 Hour": "1h", "4 Hours": "4h", "24 Hours": "24h"}
        horizon = horizon_map.get(horizon_label, "1h")

        st.sidebar.markdown("### 🧠 Model Selection")
        model_choice = st.sidebar.selectbox(
            "Pick model architecture (or Auto for best available)",
            ("Auto", "LSTM", "GRU"),
            index=0
        )

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
        st.sidebar.markdown("**🔗 Powered by:** Binance API")
        st.sidebar.markdown("**⏱ Timezone:** IST ")
        st.sidebar.caption("Built with Streamlit • Model: LSTM / GRU")

        return {
            "pair": pair,
            "horizon": horizon,
            "theme_mode": theme_mode,
            "model_choice": model_choice,
            "run": run,
            "compare": compare
        }

    
    @staticmethod
    def validate_config(pair, horizon):
        """Validate user configuration."""
        cfg = Config.MODEL_MAP.get((pair, horizon))
        if cfg is None:
            st.error("No model configured for this pair/horizon. Update Config.MODEL_MAP in the app.")
            return False
        return True


# ============================
# MAIN APP
# ============================
def main():
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="CryptoSense AI", page_icon="CPP_logo.png", layout="wide")
    
    def img_to_base64(path: str) -> str:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode()

    logo_b64 = img_to_base64("CPP_logo.png")

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:18px; margin-bottom:14px;">
            <div style="flex:0 0 auto">
                <div style="width:68px; height:68px; border-radius:14px; display:flex; align-items:center; justify-content:center;
                            box-shadow: 0 4px 14px rgba(0,0,0,0.25); overflow:hidden;">
                    <img src="data:image/png;base64,{logo_b64}" style="width:100%; height:100%; object-fit:cover;border-radius:14px" />
                </div>
            </div>
            <div style="flex:1 1 auto">
                <h1 style="margin:0; font-size:28px;">CryptoSense AI - CryptoCurrency Price Predictor</h1>
                <div style="color: #98A0AA; font-size:14px;">
                    Choose a pair and horizon to generate fast, interactive crypto forecasts.
                </div>
            </div>
            <div style="flex:0 0 auto; text-align:right;">
                <div style="font-size:12px; color:#98A0AA;">Status</div>
                <div style="font-weight:700; color:#00FFFF; margin-top:4px;">Ready</div>
            </div>
        </div>
        <hr style="border:none; height:1px; background:linear-gradient(90deg, rgba(255,255,255,0), rgba(200,200,200,0.12), rgba(255,255,255,0)); margin-bottom:18px;" />
        """,
        unsafe_allow_html=True
    )

    user_input = InputHandler.render_sidebar()

    st.markdown(
        f"<div style='color:#9AA2AA; margin-bottom:8px;'>"
        f"<strong>Pair:</strong> {user_input['pair']} &nbsp; • &nbsp; <strong>Horizon:</strong> {user_input['horizon']}"
        f"</div>",
        unsafe_allow_html=True
    )

    if not InputHandler.validate_config(user_input['pair'], user_input['horizon']):
        return

    # ---------- Predict flow ----------
    if user_input['run']:
        model, scaler, model_load_time, scaler_load_time, cfg = DataLoader.load_artifacts(
            user_input['pair'],
            user_input['horizon'],
            user_input['model_choice']
        )

        if model is None or scaler is None:
            st.warning("Model or scaler not loaded. Make sure files exist in ./models/")
            return

        with st.spinner("Preparing data and running prediction..."):
            df, lookback, n_features, yf_interval = DataLoader.fetch_and_prepare_data(
                user_input['pair'],
                user_input['horizon'],
                model,
                scaler
            )

            if df is None:
                return

            result = PredictionEngine.run_prediction(df, model, scaler, lookback, n_features)

            if result is None:
                return

            future_index = TimelineBuilder.build_future_timeline(
                df.index[-1],
                yf_interval,
                result['preds'].shape[0]
            )
            pred_series = pd.Series(result['preds'].ravel(), index=future_index)
            model_type = (cfg.get('type') if isinstance(cfg, dict) else None) or 'Model'

            latest_price = df['close'].iloc[-1]
            first_pred = pred_series.iloc[0]
            pct_change = (first_pred - latest_price) / latest_price * 100.0
            ci_low = first_pred - 1.96 * (result['resid_std'] or 0.0)
            ci_high = first_pred + 1.96 * (result['resid_std'] or 0.0)

            left_col, right_col = st.columns([3, 1], gap="large")

            with left_col:
                fig = Visualizer.plot_predictions(
                    df,
                    pred_series,
                    result['resid_std'],
                    user_input['pair'],
                    user_input['horizon'],
                    lookback=lookback,
                    label1=model_type
                )
                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.markdown("<div style='margin-bottom:8px; font-weight:700;'>Quick KPIs</div>", unsafe_allow_html=True)
                st.metric(label="Current Price", value=f"${latest_price:,.2f}")
                st.metric(label=f"Next-step Δ ({model_type})", value=f"{pct_change:+.2f}%", delta=f"${(first_pred - latest_price):.2f}")
                st.markdown(f"<div style='margin-top:8px; color:#8F99A6; font-size:13px;'>Latency: <strong>{result['predict_time']*1000:.0f} ms</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#8F99A6; font-size:13px;'>CI (95%): <strong>${ci_low:.2f}</strong> — <strong>${ci_high:.2f}</strong></div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("<div style='font-size:13px; color:#98A0AA;'>Model</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='padding:8px 10px; border-radius:8px; background:#0f1720; color:#00FFFF; font-weight:700; display:inline-block;'>{model_type}</div>", unsafe_allow_html=True)

            with st.expander("Show raw input & predictions", expanded=False):
                _, col = st.columns([4.5, 1])
                with col:
                    Visualizer.download_csv(df, pred_series, user_input['pair'], user_input['horizon'])
                st.write("Last rows of input data:")
                st.dataframe(df.tail(8))
                st.write("Predictions:")
                st.dataframe(pred_series.to_frame("pred"))

            st.divider()

            Visualizer.display_summary(
                df,
                pred_series,
                result['predict_time'],
                model_load_time,
                scaler_load_time,
                result['resid_std'],
                cfg
            )

            st.success("Prediction complete.")

    # ---------- Compare LSTM vs GRU flow ----------
    if user_input.get('compare'):
        key = (user_input['pair'], user_input['horizon'])
        pool = Config.MODEL_MAP.get(key)
        if pool is None:
            st.error("No model configs for this pair/horizon.")
            return

        lstm_cfg = pool.get('lstm') or pool.get('LSTM') or pool.get('best')
        gru_cfg = pool.get('gru') or pool.get('GRU') or pool.get('best')

        if lstm_cfg is None or gru_cfg is None:
            st.error("LSTM or GRU config missing for this pair/horizon.")
            return

        from src.modeling import ModelLoader
        
        scaler_path = lstm_cfg.get('scaler') or gru_cfg.get('scaler')
        scaler = ModelLoader.load_scaler(scaler_path)
        lstm_model = ModelLoader.load_keras_model(lstm_cfg.get('model'))
        gru_model = ModelLoader.load_keras_model(gru_cfg.get('model'))

        if lstm_model is None or gru_model is None or scaler is None:
            st.error("Failed to load models/scaler for comparison.")
            return

        example_model = lstm_model
        df, lookback, n_features, yf_interval = DataLoader.fetch_and_prepare_data(
            user_input['pair'], user_input['horizon'], example_model, scaler
        )
        if df is None:
            return

        with st.spinner("Running LSTM and GRU predictions..."):
            results = PredictionEngine.compare_models(df, lstm_model, gru_model, scaler, lookback, n_features, parallel=True)

            if not results.get('lstm') or not results.get('gru'):
                st.error("Comparison failed; one or both models returned no result.")
                return

            lstm_res = results['lstm']
            gru_res = results['gru']

            # Get actual number of predictions returned by models
            n_steps = lstm_res['preds'].shape[0]
            
            future_index = TimelineBuilder.build_future_timeline(df.index[-1], yf_interval, n_steps)
            lstm_series = pd.Series(lstm_res['preds'].ravel(), index=future_index)

            future_index2 = TimelineBuilder.build_future_timeline(df.index[-1], yf_interval, n_steps)
            gru_series = pd.Series(gru_res['preds'].ravel(), index=future_index2)

            left_col, right_col = st.columns([2.5, 1], gap="large")
            with left_col:
                fig = Visualizer.plot_predictions(
                    df,
                    lstm_series,
                    lstm_res['resid_std'],
                    user_input['pair'],
                    user_input['horizon'],
                    lookback=lookback,
                    pred_series2=gru_series,
                    resid_std2=gru_res['resid_std'],
                    label1='LSTM',
                    label2='GRU'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Models Used")
                st.write(
                    "<span style='padding:6px 12px; background:#0f1720; color:#00FFFF; "
                    "border-radius:6px; font-weight:700; margin-right:8px;'>LSTM</span>"
                    "<span style='padding:6px 12px; background:#0f1720; color:#FFA500; "
                    "border-radius:6px; font-weight:700;'>GRU</span>",
                    unsafe_allow_html=True
                )

            with right_col:
                st.markdown("<div style='font-weight:700; margin-bottom:6px;'>Comparison</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#98A0AA; font-size:15px;'>LSTM latency: <strong>{lstm_res['predict_time']*1000:.0f} ms</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#98A0AA; font-size:15px;'>GRU latency: <strong>{gru_res['predict_time']*1000:.0f} ms</strong></div>", unsafe_allow_html=True)

                common_idx = lstm_series.index.intersection(gru_series.index)
                if len(common_idx) > 0:
                    diffs = (lstm_series.loc[common_idx] - gru_series.loc[common_idx]).abs()
                    st.metric(label="MAE across Predictions:", value=f"${diffs.mean():.2f}")
                    st.write(f"Overlap steps: {len(common_idx)}")
                else:
                    st.write("No overlapping prediction timestamps to compare directly.")

                st.markdown("---")
                with st.expander("Show model-wise predictions", expanded=False):
                    st.write("LSTM predictions")
                    st.dataframe(lstm_series.to_frame("lstm_pred"))
                    st.write("GRU predictions")
                    st.dataframe(gru_series.to_frame("gru_pred"))

            st.subheader("Comparison Summary")
            st.markdown("#### Model KPIs")

            col_l, col_g = st.columns(2, gap="large")

            with col_l:
                st.markdown("#### LSTM")
                st.write(f"Latency: `{int(lstm_res['predict_time']*1000)} ms`")
                st.write(f"Steps: `{lstm_series.shape[0]}`")
                st.write(f"First Prediction: `${lstm_series.iloc[0]:.2f}`")

            with col_g:
                st.markdown("#### GRU")
                st.write(f"Latency: `{int(gru_res['predict_time']*1000)} ms`")
                st.write(f"Steps: `{gru_series.shape[0]}`")
                st.write(f"First Prediction: `${gru_series.iloc[0]:.2f}`")

            common_idx = lstm_series.index.intersection(gru_series.index)

            st.markdown("#### Model Difference")

            if len(common_idx) > 0:
                mean_abs_diff = (lstm_series.loc[common_idx] - gru_series.loc[common_idx]).abs().mean()
                st.write(f"Mean Absolute Difference across Predictions over `{len(common_idx)}` overlapping steps: `${mean_abs_diff:.2f}` USD")
            else:
                st.info("No overlapping timestamps to compare model predictions.")

            def ci_width(res):
                return (2 * 1.96 * abs(res['resid_std'])) if (res and res.get('resid_std') is not None) else np.nan

            st.markdown("#### Forecast Uncertainty")

            col_l2, col_g2 = st.columns(2, gap="large")

            with col_l2:
                st.write(f"LSTM: 95% CI Width: `{ci_width(lstm_res):.2f}` USD")

            with col_g2:
                st.write(f"GRU: 95% CI Width: `{ci_width(gru_res):.2f}` USD")
            
            st.success("Comparison Complete.")


if __name__ == "__main__":
    main()

st.markdown("""
<style>
.simple-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 12px 0;
    background: #0e1117;
    text-align: center;
    font-size: 13px;
    color: #9aa0a6;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 40px;
}
.simple-footer span {
    color: #00e6a8;
    font-weight: 500;
}
</style>

<div class="simple-footer">
    © 2025 <span>CryptoSense AI - CryptoCurrency Price Predictor</span> • Empowering Crypto Intelligence
</div>
""", unsafe_allow_html=True)

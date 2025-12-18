import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

from src.data import DataFetcher
from src.modeling import PredictionEngine, ModelLoader
from src.visualization import Visualizer
from src.utils import Config, DataLoader, TimelineBuilder

class View:
    """Handles rendering of different application views."""

    @staticmethod
    def render_header():
        """Renders the application header with logo and status."""
        def img_to_base64(path: str) -> str:
            # Check if file exists to avoid crashes
            if not Path(path).exists():
                return ""
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

    @staticmethod
    def render_market_overview(user_input):
        """Renders the idle state market overview."""
        from src.preprocessing import FeatureProcessor
        st.markdown("### 📊 Market Overview")
        with st.spinner(f"Fetching latest data for {user_input['pair']}..."):
            # Use 'Auto' model (None) just to get data
            df, _, _, _ = DataLoader.fetch_and_prepare_data(
                user_input['pair'], 
                user_input['horizon'], 
                model=None, 
                scaler=None,
                source=user_input['source']
            )
            
            if df is not None:
                # Show key metrics
                last_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100
                
                col_metric, col_chart = st.columns([1, 3])
                
                with col_metric:
                    st.metric(
                        label="Current Price", 
                        value=f"${last_close:,.2f}", 
                        delta=f"{change:+.2f} ({pct_change:+.2f}%)"
                    )
                    st.caption(f"Source: {user_input['source']}")
                    
                    st.info(
                        "👈 Select a model and click **Predict** in the sidebar to generate future forecasts.\nClick **Compare** for model comparison."
                    )

                with col_chart:
                    # Indicators Preparation
                    df_eng = None
                    if user_input.get('show_indicators'):
                        df_eng = FeatureProcessor.engineer_features(df)
                        
                    fig = Visualizer.plot_historical_data(
                        df, 
                        user_input['pair'], 
                        user_input['horizon'], 
                        indicators_df=df_eng
                    )
                    st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_prediction_flow(user_input):
        """Executes and renders the single model prediction flow."""
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
                scaler,
                source=user_input['source']
            )

            if df is None:
                return

            steps = TimelineBuilder.compute_prediction_steps(user_input['horizon'], user_input['prediction_period_hours'])
            result = PredictionEngine.run_prediction(df, model, scaler, lookback, n_features, steps=steps)

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
                # Indicators Preparation
                df_eng = None
                if user_input.get('show_indicators'):
                    from src.preprocessing import FeatureProcessor
                    df_eng = FeatureProcessor.engineer_features(df)

                fig = Visualizer.plot_predictions(
                    df,
                    pred_series,
                    result['resid_std'],
                    user_input['pair'],
                    user_input['horizon'],
                    lookback=lookback,
                    label1=model_type,
                    indicators_df=df_eng
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- PORTFOLIO SIMULATOR ---
                holdings = user_input.get('holdings', 0.0)
                if holdings > 0:
                    current_portfolio_value = latest_price * holdings
                    projected_portfolio_value = first_pred * holdings
                    pnl = projected_portfolio_value - current_portfolio_value
                    pnl_pct = (pnl / current_portfolio_value) * 100
                    
                    st.markdown("""
                    <div style="background:#1e2329; padding:16px; border-radius:12px; margin-top:10px; margin-bottom:20px; border:1px solid #363c45;">
                        <h4 style="margin:0 0 12px 0; color:#EAECEF;">💼 Portfolio Simulation Results</h4>
                        <div style="display:flex; justify-content:space-between; gap:20px;">
                            <div>
                                <div style="font-size:12px; color:#848E9C;">Current Value</div>
                                <div style="font-size:18px; font-weight:600; color:#EAECEF;">${:,.2f}</div>
                            </div>
                            <div>
                                <div style="font-size:12px; color:#848E9C;">Projected Value</div>
                                <div style="font-size:18px; font-weight:600; color:#F0B90B;">${:,.2f}</div>
                            </div>
                            <div>
                                <div style="font-size:12px; color:#848E9C;">Est. PnL</div>
                                <div style="font-size:18px; font-weight:600; color:{};">{:+.2f} ({:+.2f}%)</div>
                            </div>
                        </div>
                    </div>
                    """.format(
                        current_portfolio_value, 
                        projected_portfolio_value,
                        "#0ECB81" if pnl >= 0 else "#F6465D",
                        pnl, 
                        pnl_pct
                    ), unsafe_allow_html=True)

            with right_col:
                st.markdown("<div style='margin-bottom:8px; font-weight:700;'>Quick KPIs</div>", unsafe_allow_html=True)
                st.metric(label="Current Price", value=f"${latest_price:,.2f}")
                st.metric(label=f"Next-step Δ ({model_type})", value=f"{pct_change:+.2f}%", delta=f"${(first_pred - latest_price):.2f}")
                st.markdown(f"<div style='margin-top:8px; color:#8F99A6; font-size:13px;'>Latency: <strong>{result['predict_time']*1000:.0f} ms</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#8F99A6; font-size:13px;'>CI (95%): <strong>${ci_low:.2f}</strong> — <strong>${ci_high:.2f}</strong></div>", unsafe_allow_html=True)
                
                st.markdown("---")

                st.markdown("<div style='font-size:13px; color:#98A0AA; margin-bottom:4px;'>Model</div>", unsafe_allow_html=True)
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

    @staticmethod
    def render_comparison_flow(user_input):
        """Executes and renders the model comparison flow."""
    @staticmethod
    def render_comparison_flow(user_input):
        """Executes and renders the model comparison flow."""
        key = (user_input['pair'], user_input['horizon'])
        pool = Config.MODEL_MAP.get(key)
        used_surrogate = None
        
        # Fallback Logic
        if pool is None:
            surrogate_pair = Config.FALLBACK_CHAIN.get(user_input['pair'])
            if surrogate_pair:
                surrogate_key = (surrogate_pair, user_input['horizon'])
                pool = Config.MODEL_MAP.get(surrogate_key)
                if pool:
                    used_surrogate = surrogate_pair

        if pool is None:
            st.error("No model configs for this pair/horizon.")
            return

        lstm_cfg = pool.get('lstm') or pool.get('LSTM') or pool.get('best')
        gru_cfg = pool.get('gru') or pool.get('GRU') or pool.get('best')

        if lstm_cfg is None or gru_cfg is None:
            st.error("LSTM or GRU config missing for this pair/horizon.")
            return

        scaler_x_path = lstm_cfg.get('scaler_x') or gru_cfg.get('scaler_x')
        scaler_y_path = lstm_cfg.get('scaler_y') or gru_cfg.get('scaler_y')
        
        scaler_x = ModelLoader.load_scaler(scaler_x_path)
        scaler_y = ModelLoader.load_scaler(scaler_y_path)
        scale_bundle = {"x": scaler_x, "y": scaler_y}

        lstm_model = ModelLoader.load_keras_model(lstm_cfg.get('model'))
        gru_model = ModelLoader.load_keras_model(gru_cfg.get('model'))

        if lstm_model is None or gru_model is None or scaler_x is None or scaler_y is None:
            st.error("Failed to load models/scaler for comparison.")
            return

        example_model = lstm_model
        df, lookback, n_features, yf_interval = DataLoader.fetch_and_prepare_data(
            user_input['pair'], user_input['horizon'], example_model, scale_bundle, source=user_input['source']
        )
        if df is None:
            return

        with st.spinner("Running LSTM and GRU predictions..."):
            steps = TimelineBuilder.compute_prediction_steps(user_input['horizon'], user_input['prediction_period_hours'])
            results = PredictionEngine.compare_models(df, lstm_model, gru_model, scale_bundle, lookback, n_features, steps=steps, parallel=True)

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
                # Indicators Preparation
                df_eng = None
                if user_input.get('show_indicators'):
                    from src.preprocessing import FeatureProcessor
                    df_eng = FeatureProcessor.engineer_features(df)

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
                    label2='GRU',
                    indicators_df=df_eng
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Models Used")
                
                # Surrogate warning removed

                st.write(
                    "<span style='padding:6px 12px; background:#0f1720; color:#00FFFF; "
                    "border-radius:6px; font-weight:700; margin-right:8px;'>LSTM</span>"
                    "<span style='padding:6px 12px; background:#0f1720; color:#FFA500; "
                    "border-radius:6px; font-weight:700;'>GRU</span>",
                    unsafe_allow_html=True
                )

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
                with st.expander("Show model-wise predictions", expanded=True):
                    st.write("LSTM predictions")
                    st.dataframe(lstm_series.to_frame("lstm_pred"))
                    st.write("GRU predictions")
                    st.dataframe(gru_series.to_frame("gru_pred"))


            
            st.success("Comparison Complete.")

    @staticmethod
    def render_footer():
        """Renders the standard footer."""
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

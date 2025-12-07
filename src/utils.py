"""
Utility functions and configuration for CryptoSense AI.
Contains helper functions, caching, and constants.
"""
import pandas as pd
import streamlit as st


class Config:
    """Central configuration for models, intervals, and settings."""
    
    MODEL_MAP = {
        ("BTC-USD","1h"): {
            "best":{"model":"models/lstm_model_BTC_USD_1h.keras", "scaler":"models/scaler_BTC_USD_1h.pkl","type":"LSTM"},
            "lstm":{"model":"models/lstm_model_BTC_USD_1h.keras", "scaler":"models/scaler_BTC_USD_1h.pkl","type":"LSTM"},
            "gru":{"model":"models/gru_model_BTC_USD_1h.keras", "scaler":"models/scaler_BTC_USD_1h.pkl","type":"GRU"}
        },
        ("BTC-USD","4h"): {
            "best":{"model":"models/gru_model_BTC_USD_4h.keras", "scaler":"models/scaler_BTC_USD_4h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_BTC_USD_4h.keras", "scaler":"models/scaler_BTC_USD_4h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_BTC_USD_4h.keras", "scaler":"models/scaler_BTC_USD_4h.pkl", "type":"GRU"}
        },
        ("BTC-USD","24h"): {
            "best":{"model":"models/gru_model_BTC_USD_24h.keras", "scaler":"models/scaler_BTC_USD_24h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_BTC_USD_24h.keras", "scaler":"models/scaler_BTC_USD_24h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_BTC_USD_24h.keras", "scaler":"models/scaler_BTC_USD_24h.pkl", "type":"GRU"}
        },
        ("ETH-USD","1h"): {
            "best":{"model":"models/lstm_model_ETH_USD_1h.keras", "scaler":"models/scaler_ETH_USD_1h.pkl","type":"LSTM"},
            "lstm":{"model":"models/lstm_model_ETH_USD_1h.keras", "scaler":"models/scaler_ETH_USD_1h.pkl","type":"LSTM"},
            "gru":{"model":"models/gru_model_ETH_USD_1h.keras", "scaler":"models/scaler_ETH_USD_1h.pkl","type":"GRU"}
        },
        ("ETH-USD","4h"): {
            "best":{"model":"models/gru_model_ETH_USD_4h.keras", "scaler":"models/scaler_ETH_USD_4h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_ETH_USD_4h.keras", "scaler":"models/scaler_ETH_USD_4h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_ETH_USD_4h.keras", "scaler":"models/scaler_ETH_USD_4h.pkl", "type":"GRU"}
        },
        ("ETH-USD","24h"): {
            "best":{"model":"models/gru_model_ETH_USD_24h.keras", "scaler":"models/scaler_ETH_USD_24h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_ETH_USD_24h.keras", "scaler":"models/scaler_ETH_USD_24h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_ETH_USD_24h.keras", "scaler":"models/scaler_ETH_USD_24h.pkl", "type":"GRU"}
        },
        ("LTC-USD","1h"): {
            "best":{"model":"models/lstm_model_LTC_USD_1h.keras", "scaler":"models/scaler_LTC_USD_1h.pkl","type":"LSTM"},
            "lstm":{"model":"models/lstm_model_LTC_USD_1h.keras", "scaler":"models/scaler_LTC_USD_1h.pkl","type":"LSTM"},
            "gru":{"model":"models/gru_model_LTC_USD_1h.keras", "scaler":"models/scaler_LTC_USD_1h.pkl","type":"GRU"}
        },
        ("LTC-USD","4h"): {
            "best":{"model":"models/gru_model_LTC_USD_4h.keras", "scaler":"models/scaler_LTC_USD_4h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_LTC_USD_4h.keras", "scaler":"models/scaler_LTC_USD_4h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_LTC_USD_4h.keras", "scaler":"models/scaler_LTC_USD_4h.pkl", "type":"GRU"}
        },
        ("LTC-USD","24h"): {
            "best":{"model":"models/gru_model_LTC_USD_24h.keras", "scaler":"models/scaler_LTC_USD_24h.pkl", "type":"GRU"},
            "lstm":{"model":"models/lstm_model_LTC_USD_24h.keras", "scaler":"models/scaler_LTC_USD_24h.pkl", "type":"LSTM"},
            "gru":{"model":"models/gru_model_LTC_USD_24h.keras", "scaler":"models/scaler_LTC_USD_24h.pkl", "type":"GRU"}
        }
    }
    
    INTERVAL_MAP = {"1h":"60m", "4h":"4h", "24h":"1d"}
    HORIZON_STEPS = {"1h":1, "4h":6, "24h":24}
    LOOKBACK_SAFETY_FACTOR = 1.5
    MIN_HISTORY_POINTS = 200
    RESIDUAL_N_SAMPLES = 100


class TimelineBuilder:
    """Constructs prediction timelines from timestamps."""
    
    @staticmethod
    def build_future_timeline(last_ts, interval, num_steps):
        """Generate future timestamps for predictions."""
        delta = TimelineBuilder._parse_interval_delta(interval)
        return [last_ts + (i+1) * delta for i in range(num_steps)]
    
    @staticmethod
    def _parse_interval_delta(interval):
        """Convert interval string to timedelta."""
        if interval.endswith('m'):
            minutes = int(interval.replace('m', ''))
            return pd.Timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval.replace('h', '')) if interval != '60m' else 1
            return pd.Timedelta(hours=hours)
        else:
            return pd.Timedelta(days=1)
    
    @staticmethod
    def compute_prediction_steps(horizon: str, days: int) -> int:
        """Convert user-selected days + horizon into forecast steps."""
        horizon_map_hours = {"1h": 1, "4h": 4, "24h": 24, "1h ": 1}
        hours_per_step = horizon_map_hours.get(horizon, 1)
        total_hours = days * 24
        steps = max(1, int(total_hours / hours_per_step))
        return steps


class DataLoader:
    """Handles data fetching and preparation."""
    
    @staticmethod
    def load_artifacts(pair, horizon, model_choice="Auto"):
        """Load model and scaler artifacts."""
        from src.modeling import ModelLoader
        import time
        
        model = None
        scaler = None
        model_load_time = 0.0
        scaler_load_time = 0.0
        cfg = None

        key = (pair, horizon)
        pool = Config.MODEL_MAP.get(key)

        if pool is None:
            st.error(f"No models configured for {pair} @ {horizon}. Please update Config.MODEL_MAP.")
            return None, None, 0.0, 0.0, None

        choice = (model_choice or "Auto").strip().upper()
        if choice in ("AUTO", "BEST"):
            cfg = pool.get("best") or pool.get("BEST") or pool.get("Best")
        elif choice == "LSTM":
            cfg = pool.get("lstm") or pool.get("LSTM") or pool.get("Lstm")
        elif choice == "GRU":
            cfg = pool.get("gru") or pool.get("GRU") or pool.get("Gru")
        else:
            cfg = pool.get("best") or pool.get("lstm") or pool.get("gru")

        if cfg is None:
            for k in ("best", "lstm", "gru"):
                if pool.get(k):
                    cfg = pool.get(k)
                    break

        if cfg is None:
            st.error(f"No usable model config found for {pair} @ {horizon}.")
            return None, None, 0.0, 0.0, None

        if choice in ("LSTM", "GRU"):
            requested_key = choice.lower()
            if requested_key not in {k.lower() for k in pool.keys()}:
                best_cfg = pool.get("best")
                if best_cfg and best_cfg is not cfg:
                    st.info(f"No {choice} model for {pair}@{horizon}. Falling back to default ({best_cfg.get('type')}).")

        model_path = cfg.get("model") or cfg.get("model_path")
        scaler_path = cfg.get("scaler") or cfg.get("scaler_path")

        if model_path is None or scaler_path is None:
            st.error("Model or scaler path missing in the selected config.")
            return None, None, 0.0, 0.0, cfg

        t0 = time.time()
        model = ModelLoader.load_keras_model(model_path)
        model_load_time = time.time() - t0

        t0 = time.time()
        scaler = ModelLoader.load_scaler(scaler_path)
        scaler_load_time = time.time() - t0

        if model is None or scaler is None:
            st.error("Failed to load model or scaler. Check files and paths in Config.MODEL_MAP.")
            return model, scaler, model_load_time, scaler_load_time, cfg

        return model, scaler, model_load_time, scaler_load_time, cfg
    
    @staticmethod
    def fetch_and_prepare_data(pair, horizon, model, scaler):
        """Fetch historical data and prepare features."""
        from src.data import DataFetcher
        
        yf_interval = Config.INTERVAL_MAP.get(horizon, '1h')
        
        try:
            _, lookback, n_features = model.input_shape
            if n_features is None:
                n_features = 1
        except Exception:
            lookback = 24
            n_features = 1
        
        needed_points = max(
            int(lookback * Config.LOOKBACK_SAFETY_FACTOR) + 20,
            Config.MIN_HISTORY_POINTS
        )
        
        period_str = DataFetcher.compute_period_string(needed_points, yf_interval)
        df = DataFetcher.fetch_history(pair, period=period_str, interval=yf_interval)
        
        if df is None or df.empty:
            st.error("Failed to fetch historical data.")
            return None, None, None, None
        
        if len(df) < needed_points:
            st.warning(
                f"Fetched only {len(df)} points; model lookback={lookback} "
                f"may require more history for best accuracy."
            )
        
        df = df.tail(needed_points)
        df['close'] = df['Close']
        
        return df, lookback, n_features, yf_interval

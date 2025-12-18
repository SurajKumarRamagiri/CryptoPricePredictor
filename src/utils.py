"""
Utility functions and configuration for CryptoSense AI.
Contains helper functions, caching, and constants.
"""
import pandas as pd
import streamlit as st


class Config:
    """Central configuration for models, intervals, and settings."""
    
    # Base path for models
    BASE_PATH = "AnalysisData/dl_models"
    SCALER_PATH = "AnalysisData/dl_scalers"

    MODEL_MAP = {
        ("BTC-USD","1h"): {
            "best":{"model": f"{BASE_PATH}/BTCUSDT_1h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/BTCUSDT_1h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/BTCUSDT_1h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "lstm":{"model": f"{BASE_PATH}/BTCUSDT_1h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/BTCUSDT_1h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/BTCUSDT_1h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/BTCUSDT_1h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/BTCUSDT_1h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/BTCUSDT_1h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("BTC-USD","4h"): {
            "best":{"model": f"{BASE_PATH}/BTCUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/BTCUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/BTCUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/BTCUSDT_4h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/BTCUSDT_4h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/BTCUSDT_4h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/BTCUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/BTCUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/BTCUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("BTC-USD","24h"): {
            "best":{"model": f"{BASE_PATH}/BTCUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/BTCUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/BTCUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/BTCUSDT_24h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/BTCUSDT_24h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/BTCUSDT_24h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/BTCUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/BTCUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/BTCUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("ETH-USD","1h"): {
            "best":{"model": f"{BASE_PATH}/ETHUSDT_1h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_1h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_1h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/ETHUSDT_1h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/ETHUSDT_1h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/ETHUSDT_1h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/ETHUSDT_1h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_1h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_1h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("ETH-USD","4h"): {
            "best":{"model": f"{BASE_PATH}/ETHUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/ETHUSDT_4h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/ETHUSDT_4h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/ETHUSDT_4h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/ETHUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("ETH-USD","24h"): {
            "best":{"model": f"{BASE_PATH}/ETHUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/ETHUSDT_24h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/ETHUSDT_24h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/ETHUSDT_24h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/ETHUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/ETHUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/ETHUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("LTC-USD","1h"): {
            "best":{"model": f"{BASE_PATH}/LTCUSDT_1h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/LTCUSDT_1h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/LTCUSDT_1h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "lstm":{"model": f"{BASE_PATH}/LTCUSDT_1h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/LTCUSDT_1h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/LTCUSDT_1h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/LTCUSDT_1h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/LTCUSDT_1h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/LTCUSDT_1h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("LTC-USD","4h"): {
            "best":{"model": f"{BASE_PATH}/LTCUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/LTCUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/LTCUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/LTCUSDT_4h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/LTCUSDT_4h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/LTCUSDT_4h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/LTCUSDT_4h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/LTCUSDT_4h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/LTCUSDT_4h_GRU_scaler_y.pkl",  "type":"GRU"}
        },
        ("LTC-USD","24h"): {
            "best":{"model": f"{BASE_PATH}/LTCUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/LTCUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/LTCUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"},
            "lstm":{"model": f"{BASE_PATH}/LTCUSDT_24h_LSTM.keras", "scaler_x": f"{SCALER_PATH}/LTCUSDT_24h_LSTM_scaler_x.pkl", "scaler_y": f"{SCALER_PATH}/LTCUSDT_24h_LSTM_scaler_y.pkl", "type":"LSTM"},
            "gru":{"model": f"{BASE_PATH}/LTCUSDT_24h_GRU.keras",  "scaler_x": f"{SCALER_PATH}/LTCUSDT_24h_GRU_scaler_x.pkl",  "scaler_y": f"{SCALER_PATH}/LTCUSDT_24h_GRU_scaler_y.pkl",  "type":"GRU"}
        }
    }
    
    INTERVAL_MAP = {"1h":"60m", "4h":"4h", "24h":"1d"}
    HORIZON_STEPS = {"1h":1, "4h":6, "24h":30}
    LOOKBACK_SAFETY_FACTOR = 1.5
    MIN_HISTORY_POINTS = 200
    MIN_HISTORY_POINTS = 200
    RESIDUAL_N_SAMPLES = 100
    
    # Mapping: Target Asset -> Surrogate Asset (for model sharing)
    FALLBACK_CHAIN = {
        # ETH-Correlated (Smart Contract Platforms / High Beta)
        "SOL-USD": "ETH-USD",
        "BNB-USD": "ETH-USD",
        "ADA-USD": "ETH-USD",
        
        # LTC-Correlated (Legacy / Payment / Old Gen)
        "XRP-USD": "LTC-USD",
        "DOGE-USD": "LTC-USD",
        
        # Catch-all or Cross-cycle?
        # If LTC-USD model missing, maybe map LTC -> BTC? 
        # For now, we assume BTC/ETH/LTC models exist.
    }


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
    def compute_prediction_steps(horizon: str, total_hours: int) -> int:
        """Convert user-selected hours + horizon into forecast steps."""
        horizon_map_hours = {"1h": 1, "4h": 4, "24h": 24}
        hours_per_step = horizon_map_hours.get(horizon, 1)
        # total_hours passed directly
        steps = max(1, int(total_hours / hours_per_step))
        return steps


class DataLoader:
    """Handles data fetching and preparation."""
    
    @staticmethod
    def load_artifacts(pair, horizon, model_choice="Auto"):
        """Load model and scaler artifacts with Surrogate Fallback."""
        from src.modeling import ModelLoader
        import time
        
        model = None
        scale_bundle = None
        model_load_time = 0.0
        scaler_load_time = 0.0
        cfg = None
        used_surrogate = None

        key = (pair, horizon)
        pool = Config.MODEL_MAP.get(key)
        
        # --- FALLBACK LOGIC ---
        if pool is None:
            # Check for surrogate
            surrogate_pair = Config.FALLBACK_CHAIN.get(pair)
            if surrogate_pair:
                surrogate_key = (surrogate_pair, horizon)
                pool = Config.MODEL_MAP.get(surrogate_key)
                if pool:
                    used_surrogate = surrogate_pair
                    # Notify user (Streamlit context)
                    # st.info(f"Using correlated model ({surrogate_pair}) for {pair} analysis.")

        if pool is None:
            st.error(f"No models configured for {pair} @ {horizon} (and no surrogate found).")
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
                     # Only show info if not using surrogate, to avoid clutter
                     if not used_surrogate:
                        st.info(f"No {choice} model for {pair}@{horizon}. Falling back to default ({best_cfg.get('type')}).")

        model_path = cfg.get("model") or cfg.get("model_path")
        scaler_x_path = cfg.get("scaler_x")
        scaler_y_path = cfg.get("scaler_y")

        if model_path is None or scaler_x_path is None or scaler_y_path is None:
            st.error("Model or scaler paths missing in the selected config.")
            return None, None, 0.0, 0.0, cfg

        t0 = time.time()
        model = ModelLoader.load_keras_model(model_path)
        model_load_time = time.time() - t0

        t0 = time.time()
        scaler_x = ModelLoader.load_scaler(scaler_x_path)
        scaler_y = ModelLoader.load_scaler(scaler_y_path)
        scaler_load_time = time.time() - t0
        
        scale_bundle = {"x": scaler_x, "y": scaler_y}

        if model is None or scaler_x is None or scaler_y is None:
            st.error("Failed to load model or scalers. Check files and paths in Config.MODEL_MAP.")
            return model, scale_bundle, model_load_time, scaler_load_time, cfg
            
        # Attach surrogate info to cfg for UI display
        if used_surrogate:
            if isinstance(cfg, dict):
                cfg['surrogate'] = used_surrogate
            else:
                cfg = {'surrogate': used_surrogate} # Should be dict anyway

        return model, scale_bundle, model_load_time, scaler_load_time, cfg
    
    @staticmethod
    def fetch_and_prepare_data(pair, horizon, model, scaler, source="Binance"):
        """Fetch historical data and prepare features."""
        from src.data import DataFetcher
        
        yf_interval = Config.INTERVAL_MAP.get(horizon, '1h')
        
        # Lookback is usually 60 or determined by model input
        try:
            _, lookback, n_features = model.input_shape
            if n_features is None:
                n_features = 1
        except Exception:
            lookback = 60 # Default to 60 as per notebook
            n_features = 11
        
        needed_points = max(
            int(lookback * Config.LOOKBACK_SAFETY_FACTOR) + 60,
            Config.MIN_HISTORY_POINTS
        )
        
        period_str = DataFetcher.compute_period_string(needed_points, yf_interval)
        df = DataFetcher.fetch_history(pair, period=period_str, interval=yf_interval, source=source)
        
        if df is None or df.empty:
            st.error("Failed to fetch historical data.")
            return None, None, None, None
        
        if len(df) < needed_points:
            st.warning(
                f"Fetched only {len(df)} points; model lookback={lookback} "
                f"may require more history for best accuracy."
            )
        
        df = df.tail(needed_points)
        # Store raw close for display
        df['close'] = df['Close']
        
        return df, lookback, n_features, yf_interval

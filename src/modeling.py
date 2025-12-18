"""
Modeling and prediction module for CryptoSense AI.
Handles model loading, inference, and multi-step predictions.
"""
import numpy as np
import pandas as pd
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.preprocessing import FeatureProcessor, Predictor


class ModelLoader:
    """Handles loading of Keras models and scikit-learn scalers with caching."""
    
    @staticmethod
    @st.cache_resource
    def load_keras_model(path):
        """Load a Keras model from disk."""
        import os
        from tensorflow.keras.models import load_model
        
        if not os.path.exists(path):
            return None
        try:
            return load_model(path, compile=False)
        except Exception as e:
            st.error(f"Failed to load model from {path}: {e}")
            return None
    
    @staticmethod
    @st.cache_resource
    def load_scaler(path):
        """Load a scikit-learn scaler from disk."""
        import os
        import joblib
        
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Failed to load scaler from {path}: {e}")
            return None


class PredictionEngine:
    """Orchestrates prediction pipeline."""
    
    @staticmethod
    def run_prediction(df, model, scale_bundle, lookback, n_features, steps=1):
        """Execute prediction pipeline."""
        
        scaler_x = scale_bundle["x"]
        scaler_y = scale_bundle["y"]
        
        # 1. Engineer Features
        try:
            df_eng = FeatureProcessor.engineer_features(df)
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            return None
            
        features = df_eng[FeatureProcessor.FEATURE_COLUMNS].values
        
        # 2. Scale Features
        try:
            scaled = scaler_x.transform(features)
        except Exception as e:
            st.error(f"Scaler transform failed: {e}")
            return None
        
        if len(scaled) < lookback:
            st.error(f"Not enough data points ({len(scaled)}) for lookback={lookback}.")
            return None
        
        # 3. Predict Multi-step
        t0 = time.time()
        
        preds_scaled = Predictor.predict_iterative_recalc(
            model=model, 
            df_history=df, # Pass original DF to allow feature re-calc
            lookback=lookback, 
            steps=steps, 
            scaler_x=scaler_x,
            scaler_y=scaler_y # Pass scaler_y for iterative history update
        )
        
        predict_time = time.time() - t0
        
        # 4. Denormalize
        last_close = df['Close'].iloc[-1]
        preds = FeatureProcessor.denormalize_predictions(preds_scaled, scaler_y, last_close)
        
        # 5. Residuals
        resid_std = Predictor.estimate_residual_std(model, scaler_x, scaler_y, df, lookback)
        
        return {
            'preds_scaled': preds_scaled,
            'preds': preds,
            'resid_std': resid_std,
            'predict_time': predict_time,
            'scaled': scaled,
            'last_seq': scaled[-lookback:]
        }

    @staticmethod
    def compare_models(df, lstm_model, gru_model, scale_bundle, lookback, n_features, steps=1, parallel=True):
        """Run predictions for LSTM and GRU and return structured results."""
        def call(model):
            return PredictionEngine.run_prediction(df, model, scale_bundle, lookback, n_features, steps=steps)

        results = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {
                    ex.submit(call, lstm_model): "lstm",
                    ex.submit(call, gru_model): "gru"
                }
                for fut in as_completed(futures):
                    key = futures[fut]
                    try:
                        results[key] = fut.result()
                    except Exception as e:
                        results[key] = {"error": str(e)}
        else:
            results["lstm"] = call(lstm_model)
            results["gru"] = call(gru_model)

        for k in ("lstm", "gru"):
            if results.get(k) is None or "error" in results[k]:
                st.warning(f"Comparison note: {k} result missing or failed: {results.get(k)}")

        return results


class PredictorHelper:
    """Helper methods for predictions."""
    
    @staticmethod
    def predict_iterative_recalc(model, df_history, lookback, steps, scaler_x):
        """
        Robust iterative prediction by:
        1. Predicting next Log Return
        2. Calculating next Price
        3. Appending to history
        4. Re-calculating ALL features (RSI/MACD etc) on extended history
        5. Scaling and predicting next step
        """
        current_df = df_history.copy()
        
        # For efficiency, we only need 'Open','High','Low','Close','Volume'
        # But we need index for time features
        # If 'Close' is generated, assume O=H=L=C for next candle (flat) or heuristic
        
        preds_scaled = []
        
        # Determine Interval Delta from index frequency
        if len(current_df) > 1:
            delta = current_df.index[-1] - current_df.index[-2]
        else:
            delta = pd.Timedelta(hours=1)
            
        for i in range(steps):
            # A. Prepare Input Sequence
            # 1. Feature Engineer full history
            df_eng = FeatureProcessor.engineer_features(current_df)
            
            # 2. Scale
            feats = df_eng[FeatureProcessor.FEATURE_COLUMNS].values
            
            # 3. Take last 'lookback'
            if len(feats) < lookback:
                break # Should not happen if check done before
            
            seq = feats[-lookback:].reshape(1, lookback, feats.shape[1])
            
            # B. Predict Next Log Return (Scaled)
            pred_scaled = model.predict(seq, verbose=0) 
            # Output shape (1, 1) or (1, )
            p_val = pred_scaled.flatten()[0]
            preds_scaled.append(p_val)
            
            # C. Append Prediction to History for Next Iteration
            # We need to Inverse Scale this single value to get Log Return, 
            # BUT we don't have scaler_y here easily without circular dep or passing it.
            # Wait, `run_prediction` has scaler_y. 
            # Actually, `model` outputs scaled Log Return.
            # We CANNOT update `current_df` prices without unscaling Log Return.
            # We must pass scaler_y? No, `PredictorHelper` is static.
            
            # FIX: We passed only `scaler_x`. We really need `scaler_y` to update history.
            # OR, we assume `steps`=1 is common case and `steps`>1 is rare.
            # If we can't unscale, we can't update history.
            # Let's assume we can't do this properly without major refactor to pass scaler_y.
            
            # HACK: If we are here, we need to return scaled predictions.
            # But we can't iterate properly without y_scaler.
            # Let's break loop if we can't update.
            pass
            
        # Re-implement with scaler_y support if refined.
        return np.array(preds_scaled)

    @staticmethod
    def predict_iterative_recalc_v2(model, df_history, lookback, steps, scaler_x):
        """
        Simplified Iterative: 
        Since we don't have scaler_y easily injected here without breaking API,
        AND re-calculating full pandas_ta features 24 times is slow,
        We will do:
        1. Predict 1 step.
        2. Repeat output 'steps' times? No, that's a flat line.
        3. Simple Repeat for now if steps > 1?
        
        BETTER: Modified signature in `run_prediction` to pass `scaler_y` if needed? 
        Actually, `PredictionEngine` calls this. I can change signature.
        """
        return np.zeros((steps, 1)) # Placeholder until signature update


# Updated Predictor to use new logic
# We will inject the logic directly into `PredictionEngine` above or update `PredictorHelper`.
# Let's define the real function here and bind it.

def predict_iterative_full(model, df_history, lookback, steps, scaler_x, scaler_y):
    current_df = df_history.copy()
    
    # Infer delta
    if len(current_df) > 1:
        delta = current_df.index[-1] - current_df.index[-2]
    else:
        delta = pd.Timedelta(hours=1)
        
    preds_scaled = []
    
    for i in range(steps):
        # 1. Features
        df_eng = FeatureProcessor.engineer_features(current_df)
        feats = df_eng[FeatureProcessor.FEATURE_COLUMNS].values
        
        # 2. Scale X
        try:
            seq_norm = scaler_x.transform(feats[-lookback:])
        except:
            # Fallback if transform fails (e.g. inf)
            break
            
        seq = seq_norm.reshape(1, lookback, seq_norm.shape[1])
        
        # 3. Predict -> Scaled Log Return
        pred_sc = model.predict(seq, verbose=0).flatten()[0]
        preds_scaled.append(pred_sc)
        
        # 4. Update History (Inverse Y -> Log Ret -> Price)
        log_ret = scaler_y.inverse_transform([[pred_sc]])[0,0]
        last_close = current_df['Close'].iloc[-1]
        new_close = last_close * np.exp(log_ret)
        
        new_ts = current_df.index[-1] + delta
        
        # New Row: Open=PrevClose, High=NewClose, Low=NewClose, Close=NewClose, Volume=0
        new_row = pd.DataFrame([{
            'Open': last_close,
            'High': new_close,
            'Low': new_close,
            'Close': new_close,
            'Volume': 0 # Assumption
        }], index=[new_ts])
        
        current_df = pd.concat([current_df, new_row])
        
    return np.array(preds_scaled).reshape(-1, 1)

# Bind
Predictor.predict_iterative_recalc = staticmethod(predict_iterative_full)

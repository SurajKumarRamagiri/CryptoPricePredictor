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
    def run_prediction(df, model, scaler, lookback, n_features, steps=1):
        """Execute prediction pipeline."""
        feature_columns = FeatureProcessor.FEATURE_COLUMNS
        
        if not set(feature_columns).issubset(df.columns):
            st.error(f"Missing expected columns. Found: {df.columns.tolist()}")
            return None
        
        df_features = df[feature_columns].dropna()
        values = df_features.values
        
        try:
            scaled = scaler.transform(values)
        except Exception as e:
            st.error(f"Scaler transform failed: {e}")
            return None
        
        if len(scaled) < lookback:
            st.error(f"Not enough data points ({len(scaled)}) for lookback={lookback}.")
            return None
        
        last_seq = scaled[-lookback:]
        cur_seq = last_seq.reshape(1, lookback, scaled.shape[1])
        
        t0 = time.time()
        if steps > 1:
            preds_scaled = Predictor.predict_iterative(model, cur_seq, steps, n_features)
            # predict_iterative returns (steps, n_features), ensure 2D
        else:
            preds_scaled = Predictor.predict_single_step(model, cur_seq)
            
            if preds_scaled.ndim == 3:
                preds_scaled = preds_scaled.reshape(preds_scaled.shape[1], preds_scaled.shape[2])
            elif preds_scaled.ndim == 2:
                preds_scaled = preds_scaled.reshape(preds_scaled.shape[1], 1)
            else:
                preds_scaled = np.array([[preds_scaled.ravel()[0]]])
                
        predict_time = time.time() - t0
        
        preds = FeatureProcessor.denormalize_predictions(preds_scaled, scaler)
        resid_std = Predictor.estimate_residual_std(model, scaler, values, lookback)
        
        return {
            'preds_scaled': preds_scaled,
            'preds': preds,
            'resid_std': resid_std,
            'predict_time': predict_time,
            'scaled': scaled,
            'last_seq': last_seq
        }

    @staticmethod
    def compare_models(df, lstm_model, gru_model, scaler, lookback, n_features, steps=1, parallel=True):
        """Run predictions for LSTM and GRU and return structured results."""
        def call(model):
            return PredictionEngine.run_prediction(df, model, scaler, lookback, n_features, steps=steps)

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
    def predict_single_step(model, cur_seq):
        """Generate a single prediction step."""
        return model.predict(cur_seq, verbose=0)
    
    @staticmethod
    def predict_iterative(model, cur_seq, steps, n_features):
        """Generate multi-step predictions iteratively."""
        preds_list = []
        cur = cur_seq.copy()
        
        for i in range(steps):
            out = PredictorHelper.predict_single_step(model, cur)
            
            if out.ndim == 3:
                next_scaled = out[:, -1, :]
            elif out.ndim == 2:
                next_scaled = out[:, -1].reshape(1, 1)
            else:
                next_scaled = out.reshape(1, 1)
            
            # Store prediction
            preds_list.append(next_scaled.ravel())
            
            # Prepare input for next step
            if next_scaled.size < n_features:
                # Model predicts target only (e.g., Close), but needs full features for input
                # Impute other features from previous step
                last_feats = cur[:, -1, :].reshape(1, n_features)
                next_feats = last_feats.copy()
                
                # Update price columns if we have standard OHLCV structure
                if n_features >= 4:
                    # Index 3 is Close. Use it as new Close.
                    val = next_scaled.ravel()[0]
                    next_feats[0, 3] = val
                    
                    # Heuristics for other price columns
                    next_feats[0, 0] = last_feats[0, 3]  # Open = Prev Close
                    next_feats[0, 1] = val               # High = Close (flat candle)
                    next_feats[0, 2] = val               # Low = Close (flat candle)
                    # Volume (4) and Time features (5,6,7) copied from last step
                else:
                    # Fallback for non-standard feature sets
                    next_feats.fill(next_scaled.ravel()[0])
                
                next_scaled_reshaped = next_feats.reshape(1, 1, n_features)
            else:
                next_scaled_reshaped = next_scaled.reshape(1, 1, n_features)
            
            cur = np.concatenate([cur[:, 1:, :], next_scaled_reshaped], axis=1)
        
        return np.array(preds_list).reshape(len(preds_list), -1)


# Add predict_single_step to Predictor for backward compatibility
Predictor.predict_single_step = staticmethod(PredictorHelper.predict_single_step)
Predictor.predict_iterative = staticmethod(PredictorHelper.predict_iterative)

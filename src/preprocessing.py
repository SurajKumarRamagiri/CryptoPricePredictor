"""
Feature preprocessing and engineering module for CryptoSense AI.
Handles feature extraction, normalization, and data preparation.
"""
import numpy as np
import pandas as pd
import streamlit as st


class FeatureProcessor:
    """Handles feature extraction and normalization."""
    
    FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour', 'day_of_week', 'month']
    
    @staticmethod
    def prepare_features(df):
        """Extract and validate feature columns."""
        if not set(FeatureProcessor.FEATURE_COLUMNS).issubset(df.columns):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        df_features = df[FeatureProcessor.FEATURE_COLUMNS].dropna()
        return df_features.values
    
    @staticmethod
    def denormalize_predictions(preds_scaled, scaler):
        """Convert scaled predictions back to original price range."""
        try:
            if preds_scaled.shape[1] == 1:
                dummy = np.zeros((preds_scaled.shape[0], 8))
                dummy[:, 3] = preds_scaled.ravel()
                preds_all = scaler.inverse_transform(dummy)
                return preds_all[:, 3:4]
            else:
                preds = scaler.inverse_transform(preds_scaled)
                return preds[:, 3:4]
        except Exception as e:
            st.error(f"Denormalization failed: {e}")
            return preds_scaled


class Predictor:
    """Handles sequence creation and residual estimation."""
    
    @staticmethod
    def create_sequences(values, lookback):
        """Create sequences of fixed lookback length."""
        X = []
        for i in range(len(values) - lookback + 1):
            X.append(values[i:i+lookback])
        return np.array(X)
    
    @staticmethod
    def estimate_residual_std(model, scaler, values, lookback, n_samples=100):
        """Estimate residual std in price units for the Close column."""
        try:
            close_idx = FeatureProcessor.FEATURE_COLUMNS.index('Close')
        except Exception:
            close_idx = 3

        try:
            scaled = scaler.transform(values)
        except Exception:
            scaled = np.asarray(values, dtype=float)

        seqs = Predictor.create_sequences(scaled, lookback)
        n_seqs = len(seqs)
        if n_seqs <= 1:
            return 0.0

        m = min(n_samples, n_seqs - 1)
        seqs_to_test = seqs[-m:]

        y_true_all = scaled[lookback: lookback + n_seqs, close_idx]
        y_true_last = y_true_all[-m:]

        preds_batch = model.predict(seqs_to_test, verbose=0)

        if preds_batch.ndim == 3:
            feats = preds_batch.shape[2]
            if feats == scaled.shape[1] and close_idx < feats:
                preds_scaled = preds_batch[:, -1, close_idx]
            else:
                preds_scaled = preds_batch[:, -1, 0]
        elif preds_batch.ndim == 2:
            if preds_batch.shape[1] == 1:
                preds_scaled = preds_batch[:, 0]
            else:
                preds_scaled = preds_batch[:, -1]
        else:
            preds_scaled = preds_batch.ravel()

        m_len = len(preds_scaled)
        n_feats = scaled.shape[1]

        dummy_preds = np.zeros((m_len, n_feats))
        dummy_preds[:, close_idx] = preds_scaled
        try:
            denorm_preds = scaler.inverse_transform(dummy_preds)[:, close_idx]
        except Exception:
            denorm_preds = preds_scaled

        dummy_true = np.zeros((m_len, n_feats))
        dummy_true[:, close_idx] = y_true_last
        try:
            denorm_true = scaler.inverse_transform(dummy_true)[:, close_idx]
        except Exception:
            denorm_true = y_true_last

        mask = ~ (np.isnan(denorm_true) | np.isnan(denorm_preds))
        if mask.sum() == 0:
            return 0.0
        resid = denorm_true[mask] - denorm_preds[mask]
        if resid.size <= 1:
            return 0.0

        std_price = float(np.std(resid, ddof=1))
        return std_price

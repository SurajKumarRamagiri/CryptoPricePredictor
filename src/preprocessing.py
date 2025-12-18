"""
Feature preprocessing and engineering module for CryptoSense AI.
Handles feature extraction, normalization, and data preparation.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import streamlit as st


class FeatureProcessor:
    """Handles feature extraction and normalization."""
    
    # New Feature Set from Notebook
    FEATURE_COLUMNS = [
        'Log_Returns', 
        'Volume_Delta', 
        'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
        'RSI', 'ATR', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    ]
    
    @staticmethod
    def prepare_features(df):
        """Extract and validate feature columns using advanced engineering."""
        # Ensure we have base columns
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not set(base_cols).issubset(df.columns):
            # Try to map lower case if present
            df = df.rename(columns={c.lower(): c for c in base_cols})
            if not set(base_cols).issubset(df.columns):
                raise ValueError(f"Missing required base columns {base_cols}. Found: {df.columns.tolist()}")
        
        # Apply Engineering
        df_processed = FeatureProcessor.engineer_features(df)
        
        # Verify
        missing = [c for c in FeatureProcessor.FEATURE_COLUMNS if c not in df_processed.columns]
        if missing:
            raise ValueError(f"Feature engineering failed to create: {missing}")
            
        return df_processed[FeatureProcessor.FEATURE_COLUMNS].values
    
    @staticmethod
    def engineer_features(df):
        """
        Applies technical analysis and feature engineering.
        Matches notebook logic exactly.
        """
        df = df.copy()
        
        # 1. Targets: Log Returns
        # Formula: ln(Price_t / Price_t-1)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Time: Cyclical Encodings
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 3. Indicators: RSI, MACD, ATR, Volume Delta
        # RSI
        df['RSI'] = df.ta.rsi(length=14)
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1) 
        
        # ATR
        df['ATR'] = df.ta.atr(length=14)

        # Bollinger Bands
        bbands = df.ta.bbands(length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        
        # Volume Delta
        df['Volume_Delta'] = df['Volume'].diff()
        
        # 4. Clean Up
        df.dropna(inplace=True)
        
        return df
    
    @staticmethod
    def denormalize_predictions(preds_scaled, scaler_y, last_close):
        """
        Convert scaled LOG RETURNS back to price.
        preds_scaled: (n_steps, 1) or (n_steps,)
        scaler_y: The scaler for Log_Returns
        last_close: The actual Close price before prediction start
        """
        try:
            # 1. Inverse Transform Log Returns
            if preds_scaled.ndim == 1:
                preds_scaled = preds_scaled.reshape(-1, 1)
                
            log_returns = scaler_y.inverse_transform(preds_scaled).flatten()
            
            # 2. Reconstruct Price Path: Price_t = Price_{t-1} * exp(r_t)
            prices = []
            curr_price = last_close
            
            for ret in log_returns:
                next_price = curr_price * np.exp(ret)
                prices.append(next_price)
                curr_price = next_price
                
            return np.array(prices).reshape(-1, 1)
            
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
    def estimate_residual_std(model, scaler_x, scaler_y, df_raw, lookback, n_samples=100):
        """
        Estimate residual std in PRICE units.
        Complex because model predicts returns, but we want price error.
        """
        try:
            # 1. Prepare Features exactly as training
            df_eng = FeatureProcessor.engineer_features(df_raw)
            features = df_eng[FeatureProcessor.FEATURE_COLUMNS].values
            
            # 2. Scale Inputs
            features_scaled = scaler_x.transform(features)
            
            # 3. Create Sequences
            seqs = Predictor.create_sequences(features_scaled, lookback)
            if len(seqs) < 1: return 0.0
            
            m = min(n_samples, len(seqs))
            X_test = seqs[-m:]
            
            # 4. Get Actual Log Returns (Targets)
            # Log_Returns is column 0 in FEATURE_COLUMNS
            # We want the targets *after* the sequence
            # If seq ends at i, target is at i + lookback (which is the next step from perspective of i)
            # The last sequence ends at index -1. Its target is the Data value at index -1?
            # Wait, create_sequences: values[i:i+lookback]. 
            # If we want to predict the *next* step, we usually assume y is at i+lookback.
            # But we don't have y for the very last sequence if it's "the future".
            # Here we test on *past* data where we have ground truth.
            
            # Indices in features array corresponding to targets
            # X[k] is features[k : k+lookback]
            # y[k] is features[k+lookback] (specifically column 0)
            target_indices = range(lookback, lookback + len(X_test))
            # Ensure we don't go out of bounds
            valid_m = 0
            for idx in target_indices:
                if idx < len(features):
                    valid_m += 1
            
            if valid_m == 0: return 0.0
            
            X_test = X_test[:valid_m]
            target_indices = target_indices[:valid_m]
            
            actual_log_rets = features[target_indices, 0] # 0 is Log_Returns
            
            # 5. Predict
            preds_scaled = model.predict(X_test, verbose=0)
            preds_log_rets = scaler_y.inverse_transform(preds_scaled).flatten()
            
            # 6. Convert to Price Space Error?
            # Error in log return ~= % Error in price
            # Standard Deviation of Log Return Prediction Error * Current Price ~= Price Error Std
            
            resid_log = actual_log_rets - preds_log_rets
            std_log = np.std(resid_log)
            
            # Convert to approximate price std using last price
            last_price = df_raw['Close'].iloc[-1]
            std_price = last_price * std_log
            
            return float(std_price)

        except Exception as e:
            # print(f"Residual calc error: {e}")
            return 0.0

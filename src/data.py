"""
Data fetching and loading module for CryptoSense AI.
Handles Binance API interactions and data preprocessing.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class DataFetcher:
    """Handles fetching and preprocessing historical data from Binance API."""
    
    BINANCE_API = "https://api.binance.us/api/v3/klines"
    
    BINANCE_INTERVALS = {
        "1h": "1h",
        "4h": "4h", 
        "24h": "1d",
        "1d": "1d"
    }
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_history(pair, period, interval, source="Binance"):
        """Fetch OHLCV data from API with fallback support."""
        
        # Determine order of sources to try
        sources_to_try = [source]
        if source == "Binance":
            sources_to_try.append("Yahoo Finance")
        else:
            sources_to_try.append("Binance")
            
        df = None
        used_source = None
        errors = []

        for src in sources_to_try:
            try:
                if src == "Yahoo Finance":
                    yf_symbol = DataFetcher._convert_pair_to_yfinance(pair)
                    df = DataFetcher._fetch_from_yfinance(yf_symbol, interval, period)
                else:
                    binance_symbol = DataFetcher._convert_pair_to_binance(pair)
                    binance_interval = DataFetcher.BINANCE_INTERVALS.get(interval, "1h")
                    num_klines = DataFetcher._calculate_num_klines(period, interval)
                    df = DataFetcher._fetch_from_binance(binance_symbol, binance_interval, num_klines)
                
                if df is not None and not df.empty:
                    used_source = src
                    break
            except Exception as e:
                errors.append(f"{src}: {str(e)}")
                continue

        if df is None or df.empty:
            st.error(f"Failed to fetch data for {pair}. Errors: {'; '.join(errors)}")
            return None
        
        if used_source != source:
            st.warning(f"⚠️ Primary source ({source}) unavailable. Using data from {used_source} instead.")
        
        try:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = DataFetcher._add_time_features(df)
            return df
        except Exception as e:
             st.error(f"Error processing data from {used_source}: {e}")
             return None

    @staticmethod
    def _convert_pair_to_yfinance(pair):
        """Convert pair format to Yahoo Finance ticker (e.g. BTC-USD)."""
        # Yahoo Finance usually uses 'BTC-USD', 'ETH-USD', etc.
        return pair

    @staticmethod
    def _fetch_from_yfinance(symbol, interval, period):
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            # Map interval to yfinance allowed intervals
            # valid: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            yf_interval_map = {
                "1h": "1h",
                "4h": "1h", # yfinance doesn't support 4h, fallback to 1h is risky for model, but we can resample or just require 1h
                            # NOTE: For now, if 4h is requested, yfinance might not support it directly.
                            # Standard yfinance intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            }
            
            # Special handling for intervals not supported by YF directly (like 4h)
            # We will default to '1h' and user might notice granularity diff, or we implement resampling.
            # For simplicity in this edit, we map 24h -> 1d, 1h -> 1h. 
            # 4h is tricky. Let's use 1h and maybe later resample? 
            # Actually, existing code uses yf_interval for fetching.
            
            req_interval = "1h"
            if interval == "24h" or interval == "1d":
                req_interval = "1d"
            elif interval == "1h" or interval == "60m":
                req_interval = "1h"
            elif interval == "4h":
                # Fallback to 1h for 4h request is common if API lacks it, but model expects 4h steps?
                # The model *training* data defines what a "step" is. 
                # If we feed 1h data to a 4h model, it's bad.
                # However, yfinance doesn't return 4h candles. 
                # We will use 1h and warn used, or just return 1h and let the user beware.
                # Better: Let's stick to 1h for now or try to resample if possible.
                # Given complexity, we'll request 1h.
                req_interval = "1h" 
                
            # Period mapping
            # yfinance periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            # Our period input is like "730d".
            yf_period = "2y" # Default 730 days
            
            df = yf.download(symbol, period=yf_period, interval=req_interval, progress=False)
            
            if df is None or df.empty:
                return None
            
            # yfinance returns MultiIndex columns in new versions? 
            # Ensure flat columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # yfinance columns: Open, High, Low, Close, Adj Close, Volume
            # Rename if necessary to match required format (Open, High, Low, Close, Volume)
            # They are usually Title Case already.
            
            # Resample for 4h if needed? 
            if interval == "4h":
                # Simple resampling
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }
                df = df.resample('4h').agg(agg_dict).dropna()

            return df
        
        except Exception as e:
            st.error(f"Yahoo Finance fetch failed: {e}")
            return None
    
    @staticmethod
    def _convert_pair_to_binance(pair):
        """Convert pair format from 'BTC-USD' to 'BTCUSDT'."""
        mapping = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT",
            "LTC-USD": "LTCUSDT",
        }
        return mapping.get(pair, pair.replace("-", "") + "T")
    
    @staticmethod
    def _calculate_num_klines(period, interval):
        """Calculate the number of klines (candles) to request."""
        if period.endswith('d'):
            days = int(period[:-1])
        elif period.endswith('h'):
            days = int(period[:-1]) / 24
        else:
            days = 730
        
        interval_hours = {
            "1h": 1,
            "4h": 4,
            "24h": 24
        }
        hours_per_candle = interval_hours.get(interval, 1)
        num_candles = int(days * 24 / hours_per_candle)
        
        return min(num_candles, 1000)
    
    @staticmethod
    def _fetch_from_binance(symbol, interval, limit):
        """Fetch OHLCV data from Binance API."""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(DataFetcher.BINANCE_API, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base', 'Taker_buy_quote', 'Ignore'
            ])
            
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
            df.set_index('Open_time', inplace=True)
            df.index.name = None
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            return df
        
        except requests.exceptions.RequestException as e:
            st.error(f"Binance API request failed: {e}")
            return None
        except Exception as e:
            st.error(f"Error parsing Binance data: {e}")
            return None
    
    @staticmethod
    def _add_time_features(df):
        """Add normalized time features to the dataframe."""
        df['hour'] = df.index.hour / 23.0
        df['day_of_week'] = df.index.dayofweek / 6.0
        df['month'] = df.index.month / 11.0
        return df
    
    @staticmethod
    def compute_period_string(lookback_points, interval):
        """Estimate period string based on lookback and interval."""
        return '730d'

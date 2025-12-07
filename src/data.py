"""
Data fetching and loading module for CryptoSense AI.
Handles Binance API interactions and data preprocessing.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataFetcher:
    """Handles fetching and preprocessing historical data from Binance API."""
    
    BINANCE_API = "https://api.binance.us/api/v3/klines"
    
    BINANCE_INTERVALS = {
        "1h": "1h",
        "4h": "4h", 
        "24h": "1d"
    }
    
    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_history(pair, period, interval):
        """Fetch OHLCV data from Binance API and add time features."""
        try:
            binance_symbol = DataFetcher._convert_pair_to_binance(pair)
            binance_interval = DataFetcher.BINANCE_INTERVALS.get(interval, "1h")
            num_klines = DataFetcher._calculate_num_klines(period, interval)
            
            df = DataFetcher._fetch_from_binance(binance_symbol, binance_interval, num_klines)
            
            if df is None or df.empty:
                st.error(f"Failed to fetch data from Binance for {binance_symbol}")
                return None
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = DataFetcher._add_time_features(df)
            
            return df
        
        except Exception as e:
            st.error(f"Binance data fetch failed: {e}")
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

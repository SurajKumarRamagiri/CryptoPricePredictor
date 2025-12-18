"""
Visualization module for CryptoSense AI.
Handles chart generation, display, and data export.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


class Visualizer:
    """Handles chart generation and display."""
    
    @staticmethod
    def plot_historical_data(df, pair, horizon, lookback=48, indicators_df=None):
        """Generate interactive Plotly chart with recent historical data only."""
        from plotly.subplots import make_subplots
        
        # Handle timezone for DF
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        recent_df = df.tail(lookback)
        
        if indicators_df is not None:
             # Subplots Mode
             fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f"{pair} ({horizon})", "RSI", "MACD")
            )
             height = 700
        else:
             # Single Plot Mode
             fig = go.Figure()
             height = 500
        
        # --- Main Price Chart (Row 1) ---
        target_row = 1 if indicators_df is not None else None
        target_col = 1 if indicators_df is not None else None
        
        # Add trace helper
        def add_main(trace):
            if indicators_df is not None:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        add_main(go.Scatter(
            x=recent_df.index,
            y=recent_df['close'],
            name='Close',
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy' if indicators_df is None else None, # Remove fill in subplot to save visual noise
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))

        # --- Indicators (If enabled) ---
        if indicators_df is not None:
            ind_recent = indicators_df.tail(lookback)
            if ind_recent.index.tz is None:
                 ind_recent.index = ind_recent.index.tz_localize('UTC')
            ind_recent.index = ind_recent.index.tz_convert('Asia/Kolkata')

            # 1. Bollinger Bands (Overlay on Price)
            # 1. Bollinger Bands (Overlay on Price)
            bbu = ind_recent.get('BBU_20_2.0_2.0')
            if bbu is None:
                bbu = ind_recent.get('BBU_20_2.0')
            bbl = ind_recent.get('BBL_20_2.0_2.0')
            if bbl is None:
                bbl = ind_recent.get('BBL_20_2.0')
            if bbu is not None and bbl is not None:
                fig.add_trace(go.Scatter(x=ind_recent.index, y=bbu, name='Upper BB', line=dict(color='gray', width=1, dash='dot'), showlegend=True), row=1, col=1)
                fig.add_trace(go.Scatter(x=ind_recent.index, y=bbl, name='Lower BB', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=True), row=1, col=1)

            # 2. RSI (Row 2)
            rsi = ind_recent.get('RSI')
            if rsi is not None:
                fig.add_trace(go.Scatter(x=ind_recent.index, y=rsi, name='RSI', line=dict(color='#9467bd')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,0,0,0.5)", row=2, col=1, 
                              annotation_text="Overbought (70)", annotation_position="top right", annotation_font_color="rgba(255,0,0,0.8)")
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,0,0.5)", row=2, col=1,
                              annotation_text="Oversold (30)", annotation_position="bottom right", annotation_font_color="rgba(0,255,0,0.8)")

            # 3. MACD (Row 3)
            macd = ind_recent.get('MACD_12_26_9')
            signal = ind_recent.get('MACDs_12_26_9')
            hist = ind_recent.get('MACDh_12_26_9')
            if macd is not None:
                 fig.add_trace(go.Scatter(x=ind_recent.index, y=macd, name='MACD', line=dict(color='#17becf')), row=3, col=1)
                 fig.add_trace(go.Scatter(x=ind_recent.index, y=signal, name='Signal', line=dict(color='#e377c2')), row=3, col=1)
                 colors = np.where(hist >= 0, '#00CC96', '#EF553B')
                 fig.add_trace(go.Bar(x=ind_recent.index, y=hist, name='Hist', marker_color=colors), row=3, col=1)

        layout_args = dict(
            xaxis_title='Time' if indicators_df is None else None,
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=height,
            showlegend=False if indicators_df is not None else True, # Hide legend in complex plot to avoid clutter
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation='h', y=1.02, x=1) if indicators_df is None else None
        )
        if indicators_df is None:
            layout_args['title'] = f"{pair} — Recent Market Data ({horizon})"

        fig.update_layout(**layout_args)
        
        # Grid styling
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        
        return fig

    @staticmethod
    def plot_technical_indicators(df, pair, horizon, lookback=100):
        """Generate Technical Indicators Dashboard."""
        from plotly.subplots import make_subplots
        
        # Ensure we have data
        df = df.tail(lookback)
        
        # Create Subplots
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f"{pair} Price & Bollinger Bands", "RSI (14)", "MACD (12,26,9)")
        )
        
        # 1. Price + BBands
        # BBands cols: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        # Check if cols exist (pandas_ta naming)
        bbu = df.get('BBU_20_2.0')
        bbl = df.get('BBL_20_2.0')
        bbm = df.get('BBM_20_2.0')
        
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='white', width=1)), row=1, col=1)
        
        if bbu is not None and bbl is not None:
            fig.add_trace(go.Scatter(x=df.index, y=bbu, name='Upper BB', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bbl, name='Lower BB', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
            if bbm is not None:
                 fig.add_trace(go.Scatter(x=df.index, y=bbm, name='MA 20', line=dict(color='#FFA500', width=1)), row=1, col=1)

        # 2. RSI
        rsi = df.get('RSI')
        if rsi is not None:
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#9467bd')), row=2, col=1)
            # Add 70/30 lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
            
        # 3. MACD
        # MACD cols: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        macd = df.get('MACD_12_26_9')
        signal = df.get('MACDs_12_26_9')
        hist = df.get('MACDh_12_26_9')
        
        if macd is not None:
             fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='#17becf')), row=3, col=1)
             fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#e377c2')), row=3, col=1)
             # Histogram
             colors = np.where(hist >= 0, '#00CC96', '#EF553B')
             fig.add_trace(go.Bar(x=df.index, y=hist, name='Hist', marker_color=colors), row=3, col=1)

        fig.update_layout(
            height=800, 
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        
        return fig

    @staticmethod
    def plot_predictions(df, pred_series, resid_std, pair, horizon, lookback=60,
                        pred_series2=None, resid_std2=None, label1='LSTM', label2='GRU', indicators_df=None):
        """Generate interactive Plotly chart with recent historical data."""
        from plotly.subplots import make_subplots

        # Handle timezone for DF
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('Asia/Kolkata')

        # Handle timezone for prediction series
        if pred_series.index.tz is None:
            pred_series.index = pd.DatetimeIndex(pred_series.index).tz_localize('UTC')
        else:
            pred_series.index = pd.DatetimeIndex(pred_series.index)
        pred_series.index = pred_series.index.tz_convert('Asia/Kolkata')

        if pred_series2 is not None:
            if pred_series2.index.tz is None:
                pred_series2.index = pd.DatetimeIndex(pred_series2.index).tz_localize('UTC')
            else:
                 pred_series2.index = pd.DatetimeIndex(pred_series2.index)
            pred_series2.index = pred_series2.index.tz_convert('Asia/Kolkata')

        if indicators_df is not None:
             # Subplots Mode
             fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f"{pair} Prediction ({horizon})", "RSI", "MACD")
            )
             height = 750 # Taller for predictions
        else:
             fig = go.Figure()
             height = 500

        recent_df = df.tail(lookback)
        
        # Helper to route traces
        def add_main(trace):
            if indicators_df is not None:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        add_main(go.Scatter(
            x=recent_df.index,
            y=recent_df['close'],
            name='Close',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))

        last_ts = df.index[-1]
        last_close = df['close'].iloc[-1]

        connected_x = [last_ts] + list(pred_series.index)
        connected_y = [last_close] + list(pred_series.values)

        if resid_std is None:
            resid_std = 0.0

        upper_future = pred_series.values + 1.96 * resid_std
        lower_future = pred_series.values - 1.96 * resid_std
        upper_full = np.concatenate([[last_close], upper_future])
        lower_full = np.concatenate([[last_close], lower_future])

        customdata = np.column_stack([np.array(connected_y, dtype=float), upper_full, lower_full])
        hover_template = (
            f"{label1} Predicted: $%{{customdata[0]:.2f}}<br>"
            "Upper (95%): $%{customdata[1]:.2f}<br>"
            "Lower (95%): $%{customdata[2]:.2f}<extra></extra>"
        )

        add_main(go.Scatter(
            x=connected_x,
            y=connected_y,
            name=f'{label1} Predicted',
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=8),
            customdata=customdata,
            hovertemplate=hover_template
        ))

        if resid_std > 0:
            poly_x = np.concatenate([connected_x, connected_x[::-1]])
            poly_y = np.concatenate([upper_full, lower_full[::-1]])
            add_main(go.Scatter(
                x=poly_x,
                y=poly_y,
                fill='toself',
                fillcolor='rgba(255,127,14,0.18)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=True,
                name=f'{label1} Confidence (95%)'
            ))

        if pred_series2 is not None:
            connected_x2 = [last_ts] + list(pred_series2.index)
            connected_y2 = [last_close] + list(pred_series2.values)

            if resid_std2 is None:
                resid_std2 = 0.0

            upper_future2 = pred_series2.values + 1.96 * resid_std2
            lower_future2 = pred_series2.values - 1.96 * resid_std2
            upper_full2 = np.concatenate([[last_close], upper_future2])
            lower_full2 = np.concatenate([[last_close], lower_future2])

            customdata2 = np.column_stack([np.array(connected_y2, dtype=float), upper_full2, lower_full2])
            hover_template2 = (
                f"{label2} Predicted: $%{{customdata[0]:.2f}}<br>"
                "Upper (95%): $%{customdata[1]:.2f}<br>"
                "Lower (95%): $%{customdata[2]:.2f}<extra></extra>"
            )

            add_main(go.Scatter(
                x=connected_x2,
                y=connected_y2,
                name=f'{label2} Predicted',
                mode='lines+markers',
                line=dict(color='#2ca02c', width=2, dash='dot'),
                marker=dict(size=8, symbol='circle-open'),
                customdata=customdata2,
                hovertemplate=hover_template2
            ))

            if resid_std2 > 0:
                poly_x2 = np.concatenate([connected_x2, connected_x2[::-1]])
                poly_y2 = np.concatenate([upper_full2, lower_full2[::-1]])
                add_main(go.Scatter(
                    x=poly_x2,
                    y=poly_y2,
                    fill='toself',
                    fillcolor='rgba(44,160,44,0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=True,
                    name=f'{label2} Confidence (95%)'
                ))

        # --- Indicators (If enabled) ---
        if indicators_df is not None:
            ind_recent = indicators_df.tail(lookback)
            if ind_recent.index.tz is None:
                 ind_recent.index = ind_recent.index.tz_localize('UTC')
            ind_recent.index = ind_recent.index.tz_convert('Asia/Kolkata')

            # 1. Bollinger Bands (Overlay on Price)
            # 1. Bollinger Bands (Overlay on Price)
            bbu = ind_recent.get('BBU_20_2.0_2.0')
            if bbu is None:
                bbu = ind_recent.get('BBU_20_2.0')
            bbl = ind_recent.get('BBL_20_2.0_2.0')
            if bbl is None:
                bbl = ind_recent.get('BBL_20_2.0')
            if bbu is not None and bbl is not None:
                fig.add_trace(go.Scatter(x=ind_recent.index, y=bbu, name='Upper BB', line=dict(color='gray', width=1, dash='dot'), showlegend=True), row=1, col=1)
                fig.add_trace(go.Scatter(x=ind_recent.index, y=bbl, name='Lower BB', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=True), row=1, col=1)

            # 2. RSI (Row 2)
            rsi = ind_recent.get('RSI')
            if rsi is not None:
                fig.add_trace(go.Scatter(x=ind_recent.index, y=rsi, name='RSI', line=dict(color='#9467bd')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,0,0,0.5)", row=2, col=1,
                              annotation_text="Overbought (70)", annotation_position="top right", annotation_font_color="rgba(255,0,0,0.8)")
                fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,0,0.5)", row=2, col=1,
                              annotation_text="Oversold (30)", annotation_position="bottom right", annotation_font_color="rgba(0,255,0,0.8)")

            # 3. MACD (Row 3)
            macd = ind_recent.get('MACD_12_26_9')
            signal = ind_recent.get('MACDs_12_26_9')
            hist = ind_recent.get('MACDh_12_26_9')
            if macd is not None:
                 fig.add_trace(go.Scatter(x=ind_recent.index, y=macd, name='MACD', line=dict(color='#17becf')), row=3, col=1)
                 fig.add_trace(go.Scatter(x=ind_recent.index, y=signal, name='Signal', line=dict(color='#e377c2')), row=3, col=1)
                 colors = np.where(hist >= 0, '#00CC96', '#EF553B')
                 fig.add_trace(go.Bar(x=ind_recent.index, y=hist, name='Hist', marker_color=colors), row=3, col=1)

        layout_args = dict(
            xaxis_title='Time' if indicators_df is None else None,
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=height,
            showlegend=False if indicators_df is not None else True,
            xaxis=dict(
                tickformat='%I:%M %p<br>%d %b',
                ticklabelmode='period'
            ),
            legend=dict(
                orientation='h' if indicators_df is not None else 'v',
                x=1,
                y=1.02 if indicators_df is not None else 1,
                xanchor='right',
                yanchor='bottom' if indicators_df is not None else 'top',
                bgcolor='rgba(0,0,0,0.35)',
                bordercolor='rgba(255,255,255,0.05)',
                borderwidth=1
            )
        )

        if indicators_df is None:
            layout_args['title'] = f"{pair} — Recent History + Prediction ({horizon})"

        fig.update_layout(**layout_args)

        fig.update_xaxes(rangeslider=dict(visible=False))
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')

        return fig

    
    @staticmethod
    def display_summary(df, pred_series, predict_time, model_load_time, 
                       scaler_load_time, resid_std, cfg):
        """Display prediction summary and metrics."""
        pct_change = (pred_series.iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
        direction = "up" if pct_change > 0 else "down" if pct_change < 0 else "flat"
        
        st.subheader("Prediction Summary")
        
        if cfg:
            st.write(f"Model Architecture: **{cfg.get('type', 'Unknown')}**\n")
            st.write(f"Model Load Time: {model_load_time:.2f}s")
        
        if scaler_load_time:
            st.write(f"Scaler load time: {scaler_load_time:.2f}s")
        
        st.markdown(
            f"- Last known close: **${df['close'].iloc[-1]:.4f}**\n"
            f"- Predicted ({pred_series.index[-1].strftime('%Y-%m-%d %H:%M')}): "
            f"**${pred_series.iloc[-1]:.4f}** ({pct_change:.2f}% — **{direction}**)\n"
            f"- Prediction latency: **{predict_time*1000:.0f} ms**"
        )
        if resid_std > 0:
            st.markdown(f"- Estimated residual σ: **{resid_std:.6f}**")
    
    @staticmethod
    def display_predictions_table(pred_series):
        """Display predicted values in a table."""
        st.write("### Predicted Points")
        pred_df = pd.DataFrame({
            'timestamp': pred_series.index,
            'predicted_close': pred_series.values
        }).set_index('timestamp')
        st.dataframe(pred_df)
    
    @staticmethod
    def download_csv(df, pred_series, pair, horizon):
        """Provide CSV download option."""
        csv = pd.concat([df['close'], pred_series.rename('predicted')], axis=0).to_csv()
        st.download_button(
            "Download combined CSV",
            csv,
            file_name=f"{pair}_{horizon}_predictions.csv"
        )

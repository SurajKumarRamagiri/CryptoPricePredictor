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
    def _get_common_layout(title, height, show_legend=True, legend_kwargs=None, initial_range=None):
        """
        Returns a standardized Plotly layout dictionary optimized for responsiveness.
        """
        base_layout = dict(
            autosize=True,
            height=height,
            hovermode='x unified',
            dragmode='pan',
            showlegend=show_legend,
            margin=dict(l=0, r=0, t=130, b=0), # Tight margins for maximum width
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False
            )
        )

        if title:
            base_layout['title'] = dict(text=title, y=1, x=0.01, xanchor='left', yanchor='top')
        
        if initial_range:
            base_layout['xaxis']['range'] = initial_range
        
        if legend_kwargs:
            base_layout['legend'] = legend_kwargs
        elif show_legend:
            # Default responsive legend: Horizontal on top
            base_layout['legend'] = dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            )
            
        return base_layout

    @staticmethod
    def plot_historical_data(df, pair, horizon, lookback=100, indicators_df=None):
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
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=(f"{pair} ({horizon})", "RSI", "MACD")
            )
             height = 480
        else:
             # Single Plot Mode
             fig = go.Figure()
             height = 480
        
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
                              annotation_text="Overbought", annotation_position="top right", annotation_font_color="rgba(255,0,0,0.8)")
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
        
        # Determine Initial Zoom Range (Last ~24 points)
        # We use recent_df which is typically 'lookback' length (e.g. 48)
        # We want approx half of that or 24 points.
        # Determine Initial Zoom Range (Last ~30 points)
        # We load 'lookback' points (e.g. 100), but only zoom on last 30
        zoom_points = min(30, len(recent_df))
        zoom_start = recent_df.index[-zoom_points]
        zoom_end = recent_df.index[-1]

        # Apply Common Layout
        layout = Visualizer._get_common_layout(
            title=f"{pair} — Recent Market Data ({horizon})" if indicators_df is None else None,
            height=height,
            show_legend=(indicators_df is None),
            initial_range=[zoom_start, zoom_end]
        )
        fig.update_layout(**layout)
        
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
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold")
            
        # 3. MACD
        macd = df.get('MACD_12_26_9')
        signal = df.get('MACDs_12_26_9')
        hist = df.get('MACDh_12_26_9')
        
        if macd is not None:
             fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='#17becf')), row=3, col=1)
             fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#e377c2')), row=3, col=1)
             colors = np.where(hist >= 0, '#00CC96', '#EF553B')
             fig.add_trace(go.Bar(x=df.index, y=hist, name='Hist', marker_color=colors), row=3, col=1)
        
        # Initial Zoom (Last ~30 points)
        zoom_points = min(30, len(df))
        zoom_start = df.index[-zoom_points]
        zoom_end = df.index[-1]

        # Apply Common Layout
        layout = Visualizer._get_common_layout(
            title=None, 
            height=500,
            show_legend=False,
            initial_range=[zoom_start, zoom_end]
        )
        fig.update_layout(**layout)
        
        return fig

    @staticmethod
    def plot_predictions(df, pred_series, resid_std, pair, horizon, lookback=100,
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
                row_heights=[0.5, 0.3, 0.2],
                subplot_titles=(f"{pair} Prediction ({horizon})", "RSI", "MACD")
            )
             height = 480 
        else:
             fig = go.Figure()
             height = 480

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

        
        # Calculate Initial Zoom Range
        # Goal: Show connected prediction chain + similar amount of history
        # connected_x includes last known point + predictions.
        # Let's show ~20-25 points of history before the prediction starts.
        # recent_df = df.tail(lookback) which is typically 60 points.
        # connected_x start is recent_df.index[-1]
        
        # Get start of zoom:
        hist_steps_to_show = 30
        zoom_start_idx = max(0, len(recent_df) - hist_steps_to_show)
        zoom_start = recent_df.index[zoom_start_idx]
        
        # End of zoom is end of prediction + buffer?
        # connected_x ends at the last prediction point.
        zoom_end = connected_x[-1]

        # Apply Common Layout
        layout = Visualizer._get_common_layout(
            title=f"Recent History + Prediction ({horizon})" if indicators_df is None else None,
            height=height,
            show_legend=(indicators_df is None),
            initial_range=[zoom_start, zoom_end]
        )
        # Prediction chart specific axis formatting
        layout['xaxis']['tickformat'] = '%I:%M %p<br>%d %b'
        layout['xaxis']['ticklabelmode'] = 'period'

        fig.update_layout(**layout)
        fig.update_xaxes(rangeslider=dict(visible=False))

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
            f"**${pred_series.iloc[-1]:.4f}** ({pct_change:.2f}% - **{direction}**)\n"
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

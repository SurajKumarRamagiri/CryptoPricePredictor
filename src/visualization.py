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
    def plot_predictions(df, pred_series, resid_std, pair, horizon, lookback=60,
                        pred_series2=None, resid_std2=None, label1='LSTM', label2='GRU'):
        """Generate interactive Plotly chart with recent historical data."""

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

        fig = go.Figure()

        recent_df = df.tail(lookback)

        fig.add_trace(go.Scatter(
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

        fig.add_trace(go.Scatter(
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
            fig.add_trace(go.Scatter(
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

            fig.add_trace(go.Scatter(
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
                fig.add_trace(go.Scatter(
                    x=poly_x2,
                    y=poly_y2,
                    fill='toself',
                    fillcolor='rgba(44,160,44,0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=True,
                    name=f'{label2} Confidence (95%)'
                ))

        fig.update_layout(
            title=f"{pair} — Recent History + Prediction ({horizon})",
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=500,
            xaxis=dict(
                tickformat='%I:%M %p<br>%d %b',
                ticklabelmode='period'
            ),
            legend=dict(
                orientation='v',
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(0,0,0,0.35)',
                bordercolor='rgba(255,255,255,0.05)',
                borderwidth=1
            )
        )

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

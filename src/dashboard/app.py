import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import warnings
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

st.set_page_config(
    page_title='Forex Signal Pipeline',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    /* ── Global dark theme ── */
    html, body, [data-testid='stAppViewContainer'] {
        background-color: #0a0a0f;
        color: #e0e0e0;
    }
    [data-testid='stSidebar'] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }
    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .metric-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }
    /* ── Signal badges ── */
    .signal-buy {
        color: #00ff88;
        font-size: 2.4rem;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(0,255,136,0.5);
        letter-spacing: 3px;
    }
    .signal-sell {
        color: #ff4466;
        font-size: 2.4rem;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(255,68,102,0.5);
        letter-spacing: 3px;
    }
    .signal-hold {
        color: #ffaa00;
        font-size: 2.4rem;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(255,170,0,0.5);
        letter-spacing: 3px;
    }
    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7788ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 1px solid #2a2a4a;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    /* ── Model cards ── */
    .model-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        margin: 4px;
        height: 100%;
    }
    .model-title {
        color: #7788ff;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    .model-signal-buy  { color: #00ff88; font-size: 1.8rem; font-weight: 700; }
    .model-signal-sell { color: #ff4466; font-size: 1.8rem; font-weight: 700; }
    .model-signal-hold { color: #ffaa00; font-size: 1.8rem; font-weight: 700; }
    .model-stat {
        color: #888;
        font-size: 0.8rem;
        margin-top: 6px;
    }
    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #2a2a4a;
        margin: 24px 0;
    }
    /* ── Footer ── */
    .footer {
        color: #444;
        text-align: center;
        font-size: 0.75rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #1e1e2e;
    }
    /* ── Tab styling ── */
    .stTabs [data-baseweb='tab-list'] {
        background-color: #0f0f1a;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb='tab'] {
        color: #888;
        border-radius: 6px;
    }
    .stTabs [aria-selected='true'] {
        background-color: #1e1e3e;
        color: #7788ff;
    }
    /* ── Streamlit default overrides ── */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid='stMetricValue'] {
        color: #e0e0e0;
    }
    .stDataFrame {
        background-color: #0f0f1a;
    }
    .stAlert {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    'Close', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'roc_10',
    'sma_10', 'sma_20', 'sma_50', 'price_sma_10_ratio',
    'price_sma_20_ratio', 'price_sma_50_ratio',
    'ema_12', 'ema_26', 'ema_crossover', 'adx',
    'bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_position',
    'atr', 'atr_pct', 'daily_return', 'log_return',
    'hl_range_pct', 'gap_pct',
]

W_LSTM    = 0.60
W_SKLEARN = 0.40

@st.cache_resource
def load_models():
    models = {}
    try:
        models['lstm']        = tf.keras.models.load_model('models/lstm/EURUSD_lstm.keras')
        models['lstm_params'] = joblib.load('models/lstm/EURUSD_lstm_params.pkl')
        models['sklearn']     = joblib.load('models/sklearn/EURUSD_sklearn_best.pkl')
        models['prophet']     = joblib.load('models/prophet/EURUSD_prophet.pkl')
        with open('models/ensemble/EURUSD_ensemble_config.json') as f:
            models['config']  = json.load(f)
        return models, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def load_feature_data():
    try:
        df = pd.read_parquet('data/features/EURUSD_features.parquet')
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=300)
def fetch_live_data():
    try:
        df = yf.download(
            'EURUSD=X', period='6mo', interval='1d',
            auto_adjust=True, progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, None
    except Exception as e:
        return None, str(e)

def build_features_live(df):
    import pandas_ta_classic as ta
    df = df.copy()
    df['rsi']          = ta.rsi(df['Close'], length=14)
    macd               = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd']         = macd.iloc[:, 0]
    df['macd_signal']  = macd.iloc[:, 1]
    df['macd_hist']    = macd.iloc[:, 2]
    df['roc_10']       = ta.roc(df['Close'], length=10)
    for p in [10, 20, 50]:
        df[f'sma_{p}']              = ta.sma(df['Close'], length=p)
        df[f'price_sma_{p}_ratio']  = df['Close'] / df[f'sma_{p}']
    df['ema_12']        = ta.ema(df['Close'], length=12)
    df['ema_26']        = ta.ema(df['Close'], length=26)
    df['ema_crossover'] = df['ema_12'] - df['ema_26']
    adx                 = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['adx']           = adx.iloc[:, 0]
    bb                  = ta.bbands(df['Close'], length=20)
    bb_cols             = bb.columns.tolist()
    df['bb_upper']      = bb[bb_cols[2]]
    df['bb_mid']        = bb[bb_cols[1]]
    df['bb_lower']      = bb[bb_cols[0]]
    df['bb_width']      = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    bb_range            = df['bb_upper'] - df['bb_lower']
    df['bb_position']   = (df['Close'] - df['bb_lower']) / bb_range.replace(0, np.nan)
    df['atr']           = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['atr_pct']       = df['atr'] / df['Close']
    df['daily_return']  = df['Close'].pct_change()
    df['log_return']    = np.log(df['Close'] / df['Close'].shift(1))
    df['hl_range_pct']  = (df['High'] - df['Low']) / df['Open']
    df['gap_pct']       = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df = df.dropna()
    return df

def get_ensemble_signal(models, df):
    try:
        feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
        seq_length = models['lstm_params']['seq_length']
        X          = df[feat_cols].values
        if len(X) < seq_length:
            return None, None, None, None
        X_seq      = X[-seq_length:].reshape(1, seq_length, len(feat_cols))
        lstm_prob  = float(models['lstm'].predict(X_seq, verbose=0)[0][0])
        X_latest   = X[-1].reshape(1, -1)
        sklearn    = models['sklearn']
        if hasattr(sklearn.named_steps['classifier'], 'predict_proba'):
            sklearn_prob = float(sklearn.predict_proba(X_latest)[0][1])
        else:
            sklearn_prob = float(sklearn.predict(X_latest)[0])
        ensemble_prob = W_LSTM * lstm_prob + W_SKLEARN * sklearn_prob
        if ensemble_prob >= 0.56:
            signal = 'BUY'
        elif ensemble_prob <= 0.44:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        return signal, ensemble_prob, lstm_prob, sklearn_prob
    except Exception as e:
        return None, None, None, str(e)

def get_prophet_trend(models, df):
    try:
        prophet  = models['prophet']
        horizon  = 5
        prop_df  = pd.DataFrame({
            'ds': df.index.tz_localize(None)
                  if df.index.tz is not None else df.index,
            'y':  df['Close'].values,
        })
        future   = pd.DataFrame({
            'ds': pd.date_range(
                start=prop_df['ds'].iloc[-1],
                periods=horizon+1, freq='B'
            )
        })
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            forecast = prophet.predict(future)
        current_price = float(df['Close'].iloc[-1])
        future_price  = float(forecast['yhat'].iloc[-1])
        pct_change    = (future_price - current_price) / current_price * 100
        signal        = 'BULLISH' if future_price > current_price else 'BEARISH'
        return signal, future_price, pct_change
    except:
        return None, None, None

def plot_price_chart(df, signal, chart_days):
    df_c = df.iloc[-chart_days:]
    fig  = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=('', 'RSI (14)', 'MACD'),
    )
    fig.add_trace(go.Candlestick(
        x=df_c.index,
        open=df_c['Open'], high=df_c['High'],
        low=df_c['Low'],   close=df_c['Close'],
        name='EURUSD',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4466',
        increasing_fillcolor='#00ff88',
        decreasing_fillcolor='#ff4466',
    ), row=1, col=1)
    for sma, color in [('sma_20', '#ffaa00'), ('sma_50', '#4488ff')]:
        if sma in df_c.columns:
            fig.add_trace(go.Scatter(
                x=df_c.index, y=df_c[sma],
                name=sma.upper().replace('_', ' '),
                line=dict(color=color, width=1.5),
            ), row=1, col=1)
    if 'bb_upper' in df_c.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_c.index.to_series(), df_c.index.to_series()[::-1]]),
            y=pd.concat([df_c['bb_upper'], df_c['bb_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(100,100,255,0.07)',
            line=dict(color='rgba(100,100,255,0.3)', width=1),
            name='Bollinger Bands',
            showlegend=True,
        ), row=1, col=1)
    last_date  = df_c.index[-1]
    last_price = float(df_c['Close'].iloc[-1])
    sig_color  = '#00ff88' if signal == 'BUY' \
        else '#ff4466' if signal == 'SELL' else '#ffaa00'
    sig_symbol = 'triangle-up' if signal == 'BUY' \
        else 'triangle-down' if signal == 'SELL' else 'circle'
    fig.add_trace(go.Scatter(
        x=[last_date], y=[last_price],
        mode='markers+text',
        marker=dict(symbol=sig_symbol, size=18, color=sig_color,
                    line=dict(color='white', width=2)),
        text=[signal], textposition='top center',
        textfont=dict(color=sig_color, size=12),
        name=f'Signal: {signal}',
        showlegend=True,
    ), row=1, col=1)
    if 'rsi' in df_c.columns:
        rsi_colors = [
            '#ff4466' if v > 70 else '#00ff88' if v < 30 else '#7788ff'
            for v in df_c['rsi']
        ]
        fig.add_trace(go.Scatter(
            x=df_c.index, y=df_c['rsi'],
            name='RSI',
            line=dict(color='#aa88ff', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(170,136,255,0.05)',
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,68,102,0.08)',
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,255,136,0.08)',
                      line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash='dot', line_color='#ff4466',
                      line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash='dot', line_color='#00ff88',
                      line_width=1, row=2, col=1)
    if 'macd' in df_c.columns:
        fig.add_trace(go.Scatter(
            x=df_c.index, y=df_c['macd'],
            name='MACD', line=dict(color='#4488ff', width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_c.index, y=df_c['macd_signal'],
            name='Signal Line', line=dict(color='#ff8800', width=1.5),
        ), row=3, col=1)
        colors = ['#00ff88' if v >= 0 else '#ff4466'
                  for v in df_c['macd_hist']]
        fig.add_trace(go.Bar(
            x=df_c.index, y=df_c['macd_hist'],
            name='Histogram', marker_color=colors, opacity=0.7,
        ), row=3, col=1)
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0f0f1a',
        height=720,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h', y=1.02, x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color='#e0e0e0'),
    )
    fig.update_xaxes(gridcolor='#1e1e2e', showgrid=True)
    fig.update_yaxes(gridcolor='#1e1e2e', showgrid=True)
    return fig

def plot_forecast(df, ensemble_prob, lstm_prob, sklearn_prob, horizon=5):
    last_date    = df.index[-1]
    last_price   = float(df['Close'].iloc[-1])
    atr          = float(df['atr'].iloc[-1]) \
        if 'atr' in df.columns else last_price * 0.005
    future_dates = pd.date_range(
        start=last_date, periods=horizon+1, freq='B'
    )[1:]
    direction    = 1 if ensemble_prob >= 0.5 else -1
    magnitude    = abs(ensemble_prob - 0.5) * 2
    forecast_px  = [
        last_price + direction * magnitude * atr * (i+1)
        for i in range(horizon)
    ]
    upper        = [p + atr * 1.5 for p in forecast_px]
    lower        = [p - atr * 1.5 for p in forecast_px]
    hist_window  = df.iloc[-30:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_window.index, y=hist_window['Close'],
        name='Historical Price',
        line=dict(color='#4488ff', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[hist_window.index[-1]] + list(future_dates),
        y=[last_price] + forecast_px,
        name='Ensemble Forecast',
        line=dict(
            color='#00ff88' if direction == 1 else '#ff4466',
            width=2.5, dash='dash'
        ),
        mode='lines+markers',
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(100,136,255,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Uncertainty Band',
        showlegend=True,
    ))
    fig.add_vline(
        x=last_date,
        line_dash='dot', line_color='#444', line_width=1,
        annotation_text='Today',
        annotation_font_color='#888',
    )
    fig.add_annotation(
        x=future_dates[-1],
        y=forecast_px[-1],
        text=f'{forecast_px[-1]:.5f}',
        font=dict(
            color='#00ff88' if direction == 1 else '#ff4466',
            size=12
        ),
        showarrow=True,
        arrowcolor='#444',
        bgcolor='#1a1a2e',
        bordercolor='#2a2a4a',
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0f',
        plot_bgcolor='#0f0f1a',
        height=420,
        title=dict(
            text='5-Day Directional Forecast (LSTM 60% + Sklearn 40%)',
            font=dict(color='#7788ff', size=14),
        ),
        legend=dict(
            orientation='h', y=1.12,
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='#1e1e2e'),
        yaxis=dict(gridcolor='#1e1e2e'),
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    return fig

def render_sidebar(models, df):
    st.sidebar.markdown(
        '<p class="section-header">Pipeline Info</p>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(f"""
    **Pair:** EURUSD  
    **Timeframe:** Daily (D1)  
    **Horizon:** 5 Trading Days  
    **Last Updated:** {df.index[-1].date() if df is not None else 'N/A'}  
    """)
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="section-header">Ensemble Weights</p>',
        unsafe_allow_html=True
    )
    st.sidebar.progress(W_LSTM, text=f'LSTM: {W_LSTM*100:.0f}%')
    st.sidebar.progress(W_SKLEARN, text=f'Sklearn: {W_SKLEARN*100:.0f}%')
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p class="section-header">Holdout Performance</p>',
        unsafe_allow_html=True
    )
    st.sidebar.metric('LSTM Sharpe',    '0.1864')
    st.sidebar.metric('LSTM Win Rate',  '56.07%')
    st.sidebar.metric('LSTM Return',    '+5.87%')
    st.sidebar.metric('Ensemble Sharpe','-1.1344')
    st.sidebar.markdown('<hr>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p style="color:#444;font-size:0.75rem">'
        'Not financial advice. For educational purposes only.'
        '</p>',
        unsafe_allow_html=True
    )

def main():
    st.markdown(
        '<h1 style="color:#7788ff;letter-spacing:2px">'
        '📈 FOREX SIGNAL PIPELINE'
        '</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#888;margin-top:-10px">'
        'EURUSD Daily Bias — LSTM + Sklearn Ensemble'
        '</p>',
        unsafe_allow_html=True
    )
    models, model_err = load_models()
    if model_err:
        st.error(f'Model loading failed: {model_err}')
        st.info('Make sure models/ folder contains trained models')
        return
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        data_source = st.radio(
            'Data Source',
            ['Live Data', 'Cached Features'],
            horizontal=True,
        )
    with col2:
        chart_days = st.slider('Chart History (days)', 30, 180, 90)
    with col3:
        st.write('')
        if st.button('🔄 Refresh Signal', type='primary', use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    if data_source == 'Live Data':
        with st.spinner('Fetching live EURUSD data...'):
            raw_df, err = fetch_live_data()
        if err or raw_df is None:
            st.warning('Live data unavailable. Using cached features.')
            df, _ = load_feature_data()
        else:
            with st.spinner('Engineering features...'):
                df = build_features_live(raw_df)
    else:
        df, err = load_feature_data()
        if err:
            st.error(f'Feature data failed: {err}')
            return
    render_sidebar(models, df)
    with st.spinner('Generating ensemble signal...'):
        signal, ens_prob, lstm_prob, sklearn_prob = get_ensemble_signal(models, df)
        prophet_signal, prophet_price, prophet_pct = get_prophet_trend(models, df)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        signal_class = f'signal-{signal.lower()}' if signal else 'signal-hold'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Ensemble Signal</div>'
            f'<div class="{signal_class}">{signal or "N/A"}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c2:
        conf = f'{ens_prob*100:.1f}%' if ens_prob else 'N/A'
        conf_color = '#00ff88' if ens_prob and ens_prob > 0.56 \
            else '#ff4466' if ens_prob and ens_prob < 0.44 else '#ffaa00'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Confidence</div>'
            f'<div class="metric-value" style="color:{conf_color}">{conf}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c3:
        price = float(df['Close'].iloc[-1])
        ret   = float(df['daily_return'].iloc[-1]) * 100 \
            if 'daily_return' in df.columns else 0
        ret_color = '#00ff88' if ret >= 0 else '#ff4466'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">EURUSD Price</div>'
            f'<div class="metric-value">{price:.5f}</div>'
            f'<div style="color:{ret_color};font-size:0.85rem">{ret:+.3f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c4:
        rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None
        rsi_color = '#ff4466' if rsi and rsi > 70 \
            else '#00ff88' if rsi and rsi < 30 else '#7788ff'
        rsi_label = 'Overbought' if rsi and rsi > 70 \
            else 'Oversold' if rsi and rsi < 30 else 'Neutral'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">RSI (14)</div>'
            f'<div class="metric-value" style="color:{rsi_color}">{rsi:.1f}</div>'
            f'<div style="color:{rsi_color};font-size:0.8rem">{rsi_label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with c5:
        p_color = '#00ff88' if prophet_signal == 'BULLISH' else '#ff4466'
        p_pct   = f'{prophet_pct:+.3f}%' if prophet_pct else 'N/A'
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Prophet Trend</div>'
            f'<div class="metric-value" style="color:{p_color}">{prophet_signal or "N/A"}</div>'
            f'<div style="color:{p_color};font-size:0.8rem">{p_pct}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        '📊  Price Chart',
        '🔮  5-Day Forecast',
        '🤖  Model Breakdown',
        '📈  Performance',
    ])
    with tab1:
        st.plotly_chart(
            plot_price_chart(df, signal or 'HOLD', chart_days),
            use_container_width=True
        )
    with tab2:
        if ens_prob:
            direction_text = 'UP' if ens_prob >= 0.5 else 'DOWN'
            dir_color      = '#00ff88' if ens_prob >= 0.5 else '#ff4466'
            st.markdown(
                f'<p style="color:#888">'
                f'The ensemble model (LSTM 60% + Sklearn 40%) forecasts '
                f'EURUSD will move '
                f'<span style="color:{dir_color};font-weight:700">{direction_text}</span>'
                f' over the next 5 trading days with '
                f'<span style="color:{dir_color};font-weight:700">{ens_prob*100:.1f}%</span>'
                f' confidence.</p>',
                unsafe_allow_html=True
            )
            fig_fc = go.Figure()
            last_date    = df.index[-1]
            last_price   = float(df['Close'].iloc[-1])
            atr          = float(df['atr'].iloc[-1]) \
                if 'atr' in df.columns else last_price * 0.005
            future_dates = pd.date_range(
                start=last_date, periods=6, freq='B'
            )[1:]
            direction    = 1 if ens_prob >= 0.5 else -1
            magnitude    = abs(ens_prob - 0.5) * 2
            forecast_px  = [
                last_price + direction * magnitude * atr * (i+1)
                for i in range(5)
            ]
            upper = [p + atr * 1.5 for p in forecast_px]
            lower = [p - atr * 1.5 for p in forecast_px]
            hist  = df.iloc[-30:]
            fig_fc.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'],
                name='Historical',
                line=dict(color='#4488ff', width=2),
            ))
            fc_color = '#00ff88' if direction == 1 else '#ff4466'
            fig_fc.add_trace(go.Scatter(
                x=[hist.index[-1]] + list(future_dates),
                y=[last_price] + forecast_px,
                name='Ensemble Forecast',
                line=dict(color=fc_color, width=2.5, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, color=fc_color),
            ))
            fig_fc.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(100,136,255,0.10)',
                line=dict(color='rgba(0,0,0,0)'),
                name='ATR Uncertainty Band',
            ))
            fig_fc.add_vline(
                x=last_date,
                line_dash='dot', line_color='#555', line_width=1,
                annotation_text='  Today',
                annotation_font_color='#888',
            )
            for i, (date, price_fc) in enumerate(zip(future_dates, forecast_px)):
                fig_fc.add_annotation(
                    x=date, y=price_fc,
                    text=f'Day {i+1}<br>{price_fc:.5f}',
                    font=dict(color=fc_color, size=10),
                    showarrow=False,
                    yshift=20,
                    bgcolor='rgba(26,26,46,0.8)',
                    bordercolor='#2a2a4a',
                )
            fig_fc.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0a0a0f',
                plot_bgcolor='#0f0f1a',
                height=460,
                title=dict(
                    text='5-Day Directional Forecast — LSTM 60% + Sklearn 40%',
                    font=dict(color='#7788ff', size=14),
                ),
                legend=dict(
                    orientation='h', y=1.12,
                    bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=10, r=10, t=70, b=10),
                font=dict(color='#e0e0e0'),
                xaxis=dict(gridcolor='#1e1e2e'),
                yaxis=dict(gridcolor='#1e1e2e', tickformat='.5f'),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            fc_df = pd.DataFrame({
                'Day':           [f'Day {i+1}' for i in range(5)],
                'Date':          [d.strftime('%Y-%m-%d') for d in future_dates],
                'Forecast Price':[f'{p:.5f}' for p in forecast_px],
                'Upper Band':    [f'{p:.5f}' for p in upper],
                'Lower Band':    [f'{p:.5f}' for p in lower],
            })
            st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with tab3:
        st.markdown(
            '<p class="section-header">Individual Model Signals</p>',
            unsafe_allow_html=True
        )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            lstm_sig = 'BUY' if lstm_prob and lstm_prob >= 0.52 \
                else 'SELL' if lstm_prob and lstm_prob <= 0.44 else 'HOLD'
            lc = '#00ff88' if lstm_sig == 'BUY' \
                else '#ff4466' if lstm_sig == 'SELL' else '#ffaa00'
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-title">🧠 LSTM Neural Network</div>'
                f'<div class="model-signal-{lstm_sig.lower()}">{lstm_sig}</div>'
                f'<div class="model-stat">Probability: {lstm_prob*100:.1f}%</div>'
                f'<div class="model-stat">Weight: {W_LSTM*100:.0f}%</div>'
                f'<div class="model-stat">Holdout Win Rate: 56.07%</div>'
                f'<div class="model-stat">Holdout Return: +5.87%</div>'
                f'<div class="model-stat">Holdout Sharpe: 0.1864</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with mc2:
            sk_sig = 'BUY' if sklearn_prob and sklearn_prob >= 0.52 else 'SELL'
            sc = '#00ff88' if sk_sig == 'BUY' else '#ff4466'
            config = models.get('config', {})
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-title">⚡ Sklearn (Best Model)</div>'
                f'<div class="model-signal-{sk_sig.lower()}">{sk_sig}</div>'
                f'<div class="model-stat">Probability: {sklearn_prob*100:.1f}%</div>'
                f'<div class="model-stat">Weight: {W_SKLEARN*100:.0f}%</div>'
                f'<div class="model-stat">Selected via Optuna</div>'
                f'<div class="model-stat">30 trials across 5 models</div>'
                f'<div class="model-stat">20 walk-forward folds</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with mc3:
            pc = '#00ff88' if prophet_signal == 'BULLISH' else '#ff4466'
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-title">📅 Prophet (Informational)</div>'
                f'<div class="model-signal-{"buy" if prophet_signal == "BULLISH" else "sell"}">{prophet_signal or "N/A"}</div>'
                f'<div class="model-stat">5-day forecast: {prophet_pct:+.3f}%</div>'
                f'<div class="model-stat">Weight: 0% (excluded)</div>'
                f'<div class="model-stat">Reason: negative Sharpe on forex</div>'
                f'<div class="model-stat">Best for: seasonal business data</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-header">Technical Indicators</p>',
            unsafe_allow_html=True
        )
        ind_cols = [
            'rsi', 'macd', 'macd_signal', 'adx',
            'atr_pct', 'bb_position', 'ema_crossover', 'roc_10'
        ]
        available = [c for c in ind_cols if c in df.columns]
        if available:
            latest = df[available].iloc[-1]
            ind_data = {
                'Indicator': available,
                'Value':     [f'{v:.4f}' for v in latest.values],
                'Signal':    []
            }
            for col, val in zip(available, latest.values):
                if col == 'rsi':
                    ind_data['Signal'].append(
                        'Overbought' if val > 70 else 'Oversold' if val < 30 else 'Neutral'
                    )
                elif col == 'macd':
                    ind_data['Signal'].append('Bullish' if val > 0 else 'Bearish')
                elif col == 'ema_crossover':
                    ind_data['Signal'].append('Bullish' if val > 0 else 'Bearish')
                elif col == 'adx':
                    ind_data['Signal'].append('Strong Trend' if val > 25 else 'Weak Trend')
                elif col == 'bb_position':
                    ind_data['Signal'].append(
                        'Near Upper' if val > 0.8 else 'Near Lower' if val < 0.2 else 'Middle'
                    )
                else:
                    ind_data['Signal'].append('-')
            st.dataframe(
                pd.DataFrame(ind_data),
                use_container_width=True,
                hide_index=True,
            )
    with tab4:
        st.markdown(
            '<p class="section-header">Out-of-Sample Holdout Performance</p>',
            unsafe_allow_html=True
        )
        perf_df = pd.DataFrame({
            'Model':         ['LSTM (Primary)', 'Full Ensemble', 'Buy & Hold'],
            'Sharpe Ratio':  [0.1864, -1.1344, -0.0680],
            'Total Return':  ['+5.87%', '-2.53%', '+4.19%'],
            'Win Rate':      ['56.07%', '53.28%', 'N/A'],
            'Max Drawdown':  ['-4.88%', '-4.81%', 'N/A'],
            'Trades':        [15, 15, 1],
            'Status':        ['PRIMARY SIGNAL', 'Prophet excluded', 'Benchmark'],
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-header">Validation Methodology</p>',
            unsafe_allow_html=True
        )
        val_df = pd.DataFrame({
            'Parameter':   [
                'Validation Strategy', 'Walk-Forward Folds',
                'Training Window', 'Holdout Size',
                'Sharpe Gate Threshold', 'Optuna Trials (LSTM)',
                'Optuna Trials (Sklearn)', 'Prediction Horizon',
            ],
            'Value': [
                'Walk-Forward Validation', '20 folds',
                '500 days (~2 years)', '20% of dataset (249 days)',
                '1.0', '10 trials',
                '30 trials', '5 trading days',
            ],
        })
        st.dataframe(val_df, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="footer">'
        'Forex Signal Pipeline | Built with LSTM + Sklearn Ensemble | '
        'MLflow + DVC + GitHub Actions + Docker | '
        'For educational purposes only. Not financial advice.'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# ==========================================
# STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Money Line Scanner", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Daily Money Line Scanner")
st.markdown("Track trend flips and structural momentum confluences in real-time.")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Scanner Settings")

bullish_thresh = st.sidebar.slider("Bullish Threshold (Bright Green)", 1, 5, 3)
bearish_thresh = st.sidebar.slider("Bearish Threshold (Bright Red)", -5, -1, -3)

default_tickers = "BTC-USD, SOL-USD, SUI-USD, AAPL, NVDA"
ticker_input = st.sidebar.text_area("Watchlist Tickers (comma separated)", default_tickers)
watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# ==========================================
# NATIVE MATHEMATICAL INDICATOR ENGINES
# ==========================================
def calculate_money_line(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    close_ser = df['Close']
    high_ser = df['High']
    low_ser = df['Low']
    vol_ser = df['Volume']

    # 1. Core EMA 20 Filter
    df['EMA_20'] = close_ser.ewm(span=20, adjust=False).mean()

    # 2. RSI 14 Momentum
    delta = close_ser.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    ema_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26, 9)
    ema12 = close_ser.ewm(span=12, adjust=False).mean()
    ema26 = close_ser.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # 4. Volume SMA 20
    df['Vol_SMA20'] = vol_ser.rolling(window=20).mean()

    # 5. Native SuperTrend Engine (7, 3.0)
    hl = high_ser - low_ser
    hc = (high_ser - close_ser.shift()).abs()
    lc = (low_ser - close_ser.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/7, adjust=False).mean()
    
    hl2 = (high_ser + low_ser) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    cl_arr = close_ser.to_numpy().copy()
    ub_arr = upperband.to_numpy().copy()
    lb_arr = lowerband.to_numpy().copy()
    dir_arr = np.ones(len(df))
    
    for i in range(1, len(df)):
        if cl_arr[i] > ub_arr[i-1]:
            dir_arr[i] = 1
        elif cl_arr[i] < lb_arr[i-1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i-1]
            if dir_arr[i] == 1 and lb_arr[i] < lb_arr[i-1]:
                lb_arr[i] = lb_arr[i-1]
            if dir_arr[i] == -1 and ub_arr[i] > ub_arr[i-1]:
                ub_arr[i] = ub_arr[i-1]
                
    df['SuperTrend_Dir'] = dir_arr

    # Confluence Scoring Loop
    scores = []
    for i in range(len(df)):
        if pd.isna(df['EMA_20'].iloc[i]) or pd.isna(df['RSI'].iloc[i]) or pd.isna(df['SuperTrend_Dir'].iloc[i]):
            scores.append(0)
            continue
            
        score = 0
        score += 1 if df['SuperTrend_Dir'].iloc[i] == 1 else -1
        score += 1 if cl_arr[i] > df['EMA_20'].iloc[i] else -1
        score += 1 if df['RSI'].iloc[i] > 50 else -1
        score += 1 if df['MACD_Hist'].iloc[i] > 0 else -1
        if vol_ser.iloc[i] > df['Vol_SMA20'].iloc[i]:
            score += 1 if cl_arr[i] > df['EMA_20'].iloc[i] else -1
            
        scores.append(score)
        
    df['Money_Line_Score'] = scores

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# Set modern, wide page layout
st.set_page_config(page_title="Alpha Scanner Engine", layout="wide")

st.title("🛡️ Alpha Engine: Multi-Asset Momentum Scanner")
st.markdown("Real-time Game Theory & Technical Confluence Tracking dashboard.")

# ==============================================================================
# 1. MASTER WATCHLIST CONFIGURATION
# ==============================================================================
default_tickers = [
    # --- TECH, AI & QUANTUM EQUITIES ---
    "MRVL", "PLUG", "RGTI", "IREN", "QBTS", "CRWV", "MSTR", "CIFR", "IONQ", "HOOD", 
    "CLSK", "WULF", "QUBT", "CORZ", "NBIS", "BE", "TSM", "CBRS", "HUT", "TTWO", 
    "CAT", "MU", "META", "TSLA", "AVGO", "MSFT", "GOOGL", "AAPL", "AMZN", "HON", 
    "NVDA", "ORCL", "CRM", "PLTR", "VVV", "MET", "RARE",

    # --- MAJOR & MID-CAP CRYPTOCURRENCIES ---
    "BTC-USD", "ETH-USD", "BNB-USD", "ZEC-USD", "XMR-USD", "SOL-USD", "QNT-USD", 
    "LTC-USD", "DASH-USD", "LINK-USD", "INJ-USD", "ICP-USD", "NEAR-USD", "TON-USD", 
    "XRP-USD", "SUI-USD", "AKT-USD", "RAY-USD", "WLD-USD", "ONDO-USD", "TRX-USD", 
    "XLM-USD", "DOGE-USD", "POL-USD", "JUP-USD",

    # --- SPECULATIVE CRYPTO & MEME TOKENS ---
    "WIF-USD", "BONK-USD", "SHIB-USD", "PEPE-USD", "HYPE-USD", "DEEP-USD", 
    "BLUEF-USD", "PUMP-USD", "FARTCOIN-USD", "ALCH-USD", "ARC-USD", "PNUT-USD", 
    "USELESS-USD", "PENGU-USD", "UFD-USD", "SPX6900-USD",

    # --- GLOBAL COMMODITIES (Futures & Tracking ETFs) ---
    "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "LIT",

    # --- THEMATIC & SECTOR ETFs ---
    "GLTR", "PALL", "REMX", "SIL", "BOTZ", "IGV", "AIQU", "REXC"
]

# ==============================================================================
# 2. SIDEBAR INTERFACE & STRING PARSING FIX
# ==============================================================================
st.sidebar.header("Scanner Controls")

# Convert Python list into a clean, comma-separated string for Streamlit text area
default_tickers_string = ", ".join(default_tickers)
ticker_input = st.sidebar.text_area("Watchlist Tickers (comma separated)", default_tickers_string, height=250)

# Parse inputs accurately and remove structural artifacts
watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Timeframe Selection
lookback_period = st.sidebar.selectbox("Lookback Window", ["6mo", "3mo", "1y"], index=0)

# ==============================================================================
# 3. NATIVE MATHEMATICAL INDICATOR ENGINE
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 30:
        return None
        
    # --- 1. RSI (Welles Wilder Smoothing) ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- 2. MACD Engine ---
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # --- 3. ADX Engine (Directional Trend Strength) ---
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    
    df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = df['TR'].ewm(com=13, adjust=False).mean()
    plus_dm_smooth = df['+DM'].ewm(com=13, adjust=False).mean()
    minus_dm_smooth = df['-DM'].ewm(com=13, adjust=False).mean()
    
    df['+DI'] = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
    df['-DI'] = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    dx = 100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-10)
    df['ADX'] = dx.ewm(com=13, adjust=False).mean()
    
    # --- 4. SuperTrend Engine ---
    df['ATR'] = df['TR'].ewm(com=13, adjust=False).mean()
    multiplier = 3
    df['Upperband'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['ATR'])
    df['Lowerband'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['ATR'])
    
    upperbands = df['Upperband'].values
    lowerbands = df['Lowerband'].values
    closes = df['Close'].values
    in_trend = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if closes[i-1] > lowerbands[i-1]:
            lowerbands[i] = max(lowerbands[i], lowerbands[i-1])
        if closes[i-1] < upperbands[i-1]:
            upperbands[i] = min(upperbands[i], upperbands[i-1])
            
        if closes[i] > upperbands[i-1]:
            in_trend[i] = 1
        elif closes[i] < lowerbands[i-1]:
            in_trend[i] = -1
        else:
            in_trend[i] = in_trend[i-1]
            
    df['Trend'] = in_trend
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    return df

# =================================================================

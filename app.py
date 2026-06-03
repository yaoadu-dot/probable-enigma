import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
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

# Allow manual tuning of thresholds if needed
bullish_thresh = st.sidebar.slider("Bullish Threshold (Bright Green)", 1, 5, 3)
bearish_thresh = st.sidebar.slider("Bearish Threshold (Bright Red)", -5, -1, -3)

# Default Watchlist (Users can add more tickers right from the web UI)
default_tickers = "BTC-USD, SOL-USD, SUI-USD, AAPL, NVDA"
ticker_input = st.sidebar.text_area("Watchlist Tickers (comma separated)", default_tickers)
watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# ==========================================
# DATA ENGINE & CALCULATION FUNCTIONS
# ==========================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour to prevent API spamming
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="100d", interval="1d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def calculate_money_line(df):
    # 1. SuperTrend
    st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
    df['SuperTrend_Dir'] = st_df.iloc[:, 1] if st_df is not None else 0

    # 2. Moving Average & Momentum Indicators
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx_df.iloc[:, 0] if adx_df is not None else 0

    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD_Hist'] = macd_df.iloc[:, 1] if macd_df is not None else 0
    df['Vol_SMA20'] = ta.sma(df['Volume'], length=20)

    # Scoring Execution Loop
    scores = []
    for i in range(len(df)):
        if pd.isna(df['EMA_20'].iloc[i]) or pd.isna(df['RSI'].iloc[i]):
            scores.append(0)
            continue
        score = 0
        if df['SuperTrend_Dir'].iloc[i] == 1: score += 1
        else: score -= 1
        if df['Close'].iloc[i] > df['EMA_20'].iloc[i]: score += 1
        else: score -= 1
        if df['RSI'].iloc[i] > 50: score += 1
        else: score -= 1
        if df['MACD_Hist'].iloc[i] > 0: score += 1
        else: score -= 1
        if df['Volume'].iloc[i] > df['Vol_SMA20'].iloc[i]:
            score += 1 if df['Close'].iloc[i] > df['EMA_20'].iloc[i] else -1
            
        scores.append(score)
        
    df['Money_Line_Score'] = scores
    return df

# ==========================================
# DASHBOARD EXECUTION
# ==========================================
if st.sidebar.button("🔄 Force Refresh Scanner"):
    st.cache_data.clear()

st.subheader(f"System Snapshot — {datetime.now().strftime('%Y-%m-%d %H:%M')} GMT")

# Create a grid layout for tickers
cols = st.columns(3)
col_idx = 0

for ticker in watchlist:
    df = fetch_data(ticker)
    if df is None or len(df) < 3:
        st.warning(f"Could not load data for {ticker}")
        continue
        
    df = calculate_money_line(df)
    
    # Analyze the last two finalized daily candles
    prev_score = int(df['Money_Line_Score'].iloc[-3])
    curr_score = int(df['Money_Line_Score'].iloc[-2])
    last_price = float(df['Close'].iloc[-2])
    rsi_val = float(df['RSI'].iloc[-2])
    
    # Build status string and UI styling parameters
    if curr_score >= bullish_thresh and prev_score < bullish_thresh:
        status_text = "🟢 BULLISH FLIP"
        card_bg = "#d4edda"
        text_color = "#155724"
    elif curr_score <= bearish_thresh and prev_score > bearish_thresh:
        status_text = "🔴 BEARISH FLIP"
        card_bg = "#f8d7da"
        text_color = "#721c24"
    elif curr_score >= bullish_thresh:
        status_text = "Sustained Bullish State"
        card_bg = "#e2f0d9"
        text_color = "#2e7d32"
    elif curr_score <= bearish_thresh:
        status_text = "Sustained Bearish State"
        card_bg = "#fce4d6"
        text_color = "#c65911"
    else:
        status_text = "Neutral / Low Conviction"
        card_bg = "#f8f9fa"
        text_color = "#6c757d"

    # Render individual asset cards inside the responsive grid
    with cols[col_idx % 3]:
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
                <h3 style="margin: 0; color: #333;">{ticker}</h3>
                <p style="margin: 5px 0; font-size: 1.2em; font-weight: bold; color: {text_color};">{status_text}</p>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">
                <table style="width:100%; font-size: 0.9em; color: #333;">
                    <tr><td><b>Close Price:</b></td><td style="text-align:right;">${last_price:,.2f}</td></tr>
                    <tr><td><b>Current Score:</b></td><td style="text-align:right; font-weight:bold;">{curr_score}</td></tr>
                    <tr><td><b>Previous Score:</b></td><td style="text-align:right;">{prev_score}</td></tr>
                    <tr><td><b>RSI (14d):</b></td><td style="text-align:right;">{rsi_val:.1f}</td></tr>
                </table>
            </div>
            """, 
            unsafe_allowed_html=True
        )
    col_idx += 1

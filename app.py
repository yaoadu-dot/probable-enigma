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
# 1. PURIFIED MASTER WATCHLIST (OMITTED DELISTED/BUGGY CRYPTO PAIRS)
# ==============================================================================
default_tickers = [
    # --- TECH, AI & QUANTUM EQUITIES ---
    "MRVL", "PLUG", "RGTI", "IREN", "QBTS", "CRWV", "MSTR", "CIFR", "IONQ", "HOOD", 
    "CLSK", "WULF", "QUBT", "CORZ", "NBIS", "BE", "TSM", "CBRS", "HUT", "TTWO", 
    "CAT", "MU", "META", "TSLA", "AVGO", "MSFT", "GOOGL", "AAPL", "AMZN", "HON", 
    "NVDA", "ORCL", "CRM", "PLTR", "VVV", "MET", "RARE",

    # --- HIGH-LIQUIDITY VERIFIED CRYPTOCURRENCIES ---
    "BTC-USD", "ETH-USD", "BNB-USD", "ZEC-USD", "XMR-USD", "SOL-USD", "QNT-USD", 
    "LTC-USD", "DASH-USD", "LINK-USD", "INJ-USD", "ICP-USD", "NEAR-USD", 
    "XRP-USD", "XLM-USD", "DOGE-USD", "JUP-USD", "WIF-USD", "BONK-USD", "SHIB-USD",

    # --- GLOBAL COMMODITIES ---
    "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "LIT",

    # --- THEMATIC & SECTOR ETFs ---
    "GLTR", "PALL", "REMX", "SIL", "BOTZ", "IGV", "AIQU", "REXC"
]

# ==============================================================================
# 2. SIDEBAR INTERFACE & STRING PARSING
# ==============================================================================
st.sidebar.header("Scanner Controls")

default_tickers_string = ", ".join(default_tickers)
ticker_input = st.sidebar.text_area("Watchlist Tickers (comma separated)", default_tickers_string, height=250)

# Parse inputs accurately
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

# ==============================================================================
# 4. CONFLUENCE SCORING SYSTEM (-4 to +4)
# ==============================================================================
def compute_confluence_score(df):
    row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else row
    
    def calculate_single_score(r):
        s1 = 1 if r['Close'] > r['EMA20'] else -1
        s2 = 1 if r['RSI'] > 50 else -1
        s3 = 1 if r['MACD_Hist'] > 0 else -1
        s4 = 1 if r['Trend'] == 1 else -1
        return s1 + s2 + s3 + s4

    return calculate_single_score(row), calculate_single_score(prev_row)

# ==============================================================================
# 5. ROBUST TICKER-ISOLATED BATCH ENGINE WITH ITERATIVE FALLBACK
# ==============================================================================
processed_data = []
failed_assets = []

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

with st.spinner(f"Running secure matrix scan for {len(watchlist)} assets..."):
    try:
        # Initial primary batch download request
        batch_data = yf.download(watchlist, period=lookback_period, session=session, group_by='ticker', progress=False)
        
        if batch_data.empty:
            st.warning("Primary batch download failed. Dropping back to iterative individual processing channel...")
            available_tickers = []
        else:
            if isinstance(batch_data.columns, pd.MultiIndex):
                available_tickers = batch_data.columns.get_level_values(0).unique()
            else:
                available_tickers = [watchlist] if len(watchlist) == 1 else []

        for ticker in watchlist:
            hist = None
            
            # If the ticker was successfully obtained in bulk
            if ticker in available_tickers:
                try:
                    if isinstance(batch_data.columns, pd.MultiIndex):
                        hist = batch_data[ticker].copy()
                    else:
                        hist = batch_data.copy()
                except Exception:
                    hist = None

            # Fallback Routine: If bulk failed/skipped, query yahoo individually
            if hist is None or hist.dropna(subset=['Close']).empty:
                try:
                    hist = yf.Ticker(ticker).history(period=lookback_period)
                except Exception as individual_err:
                    failed_assets.append({"Asset": ticker, "Reason": f"Direct API Reject: {str(individual_err)}"})
                    continue

            try:
                hist = hist.dropna(subset=['Close'])
                if hist.empty or len(hist) < 30:
                    failed_assets.append({"Asset": ticker, "Reason": f"Insufficient history rows ({len(hist)})"})
                    continue
                    
                hist = calculate_indicators(hist)
                if hist is None:
                    failed_assets.append({"Asset": ticker, "Reason": "Engine calculation breakdown"})
                    continue
                    
                current_score, previous_score = compute_confluence_score(hist)
                last_row = hist.iloc[-1]
                
                processed_data.append({
                    "Asset": ticker,
                    "Current Score": current_score,
                    "Previous Score": previous_score,
                    "Price": round(last_row['Close'], 4),
                    "RSI (14d)": round(last_row['RSI'], 1),
                    "ADX (14d)": round(last_row['ADX'], 1),
                    "Trend Status": "🟢 Bullish" if last_row['Trend'] == 1 else "🔴 Bearish"
                })
            except Exception as parse_err:
                failed_assets.append({"Asset": ticker, "Reason": f"Parsing Error: {str(parse_err)}"})
                continue
                
    except Exception as general_err:
        st.error(f"Core Matrix Download Exception: {str(general_err)}")

# ==============================================================================
# 6. MODERN DASHBOARD VISUALIZATION
# ==============================================================================
if processed_data:
    scan_df = pd.DataFrame(processed_data)
    scan_df['Score Delta'] = scan_df['Current Score'] - scan_df['Previous Score']
    scan_df = scan_df.sort_values(by="Current Score", ascending=False)
    
    st.markdown("### 📊 Active Market Watchlist")
    st.dataframe(
        scan_df[["Asset", "Current Score", "Previous Score", "Price", "RSI (14d)", "ADX (14d)", "Trend Status"]],
        width="stretch",
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Scanner Alerts: High-Conviction Breakouts")
    
    high_conviction = scan_df[(scan_df['Current Score'] == 4) & (scan_df['ADX (14d)'] >= 25.0)]
    fake_outs = scan_df[(scan_df['Current Score'] == 4) & (scan_df['ADX (14d)'] < 15.0)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🚀 Validated Breakouts (ADX ≥ 25)")
        if not high_conviction.empty:
            for _, asset_row in high_conviction.iterrows():
                st.success(f"**{asset_row['Asset']}** | Score: 4 | ADX: {asset_row['ADX (14d)']} | RSI: {asset_row['RSI (14d)']}")
        else:
            st.info("No breakouts detected.")
            
    with col2:
        st.markdown("#### ⚠️ Range Grinds (ADX < 15)")
        if not fake_outs.empty:
            for _, asset_row in fake_outs.iterrows():
                st.warning(f"**{asset_row['Asset']}** | Score: 4 | ADX: {asset_row['ADX (14d)']} | RSI: {asset_row['RSI (14d)']}")
        else:
            st.info("No range traps detected.")

else:
    st.error("🚨 Critical Failure: Data arrived, but no assets could be parsed.")
    if failed_assets:
        st.markdown("### 🔍 Engine Diagnostic Logs")
        st.dataframe(pd.DataFrame(failed_assets), width="stretch", hide_index=True)

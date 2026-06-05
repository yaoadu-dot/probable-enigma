import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# App Config
st.set_page_config(page_title="Alpha Scanner", layout="wide")

st.title("🛡️ Alpha Engine: Multi-Asset Scanner")
st.markdown("Real-time Technical Confluence Tracking.")

# 1. WATCHLIST (Verified Tickers)
default_tickers = [
    "MRVL", "PLUG", "RGTI", "IREN", "QBTS", "CRWV", "MSTR", "CIFR", "IONQ", "HOOD", 
    "CLSK", "WULF", "QUBT", "CORZ", "NBIS", "BE", "TSM", "CBRS", "HUT", "TTWO", 
    "CAT", "MU", "META", "TSLA", "AVGO", "MSFT", "GOOGL", "AAPL", "AMZN", "HON", 
    "NVDA", "ORCL", "CRM", "PLTR", "VVV", "MET", "RARE", "BTC-USD", "ETH-USD", 
    "BNB-USD", "ZEC-USD", "XMR-USD", "SOL-USD", "QNT-USD", "LTC-USD", "DASH-USD", 
    "LINK-USD", "INJ-USD", "ICP-USD", "NEAR-USD", "XRP-USD", "XLM-USD", "DOGE-USD", 
    "JUP-USD", "WIF-USD", "BONK-USD", "SHIB-USD", "GC=F", "SI=F", "PL=F", "PA=F", 
    "CL=F", "BZ=F", "HG=F", "LIT", "GLTR", "PALL", "REMX", "SIL", "BOTZ", "IGV", 
    "AIQU", "REXC"
]

# 2. SIDEBAR
st.sidebar.header("Controls")
t_input = st.sidebar.text_area("Tickers", ", ".join(default_tickers), height=200)
watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]
lookback = st.sidebar.selectbox("Window", ["6mo", "3mo", "1y"], index=0)

# 3. MATH ENGINE
def calculate_indicators(df):
    if len(df) < 30: return None
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_g = gain.ewm(com=13, adjust=False).mean()
    ema_l = loss.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_g / (ema_l + 1e-10))))
    # MACD
    e12, e26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = (e12 - e26) - (e12 - e26).ewm(span=9).mean()
    # ADX
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ADX'] = (100 * (df['High'].diff().clip(lower=0) / df['TR'].ewm(com=13).mean())).ewm(com=13).mean()
    # Trend
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['Trend'] = np.where(df['Close'] > df['EMA20'], 1, -1)
    return df

def get_score(df):
    r = df.iloc[-1]
    s = (1 if r['Close'] > r['EMA20'] else -1) + (1 if r['RSI'] > 50 else -1) + (1 if r['MACD_Hist'] > 0 else -1) + (1 if r['Trend'] == 1 else -1)
    return s

# 4. DATA LOGIC
processed, failed = [], []
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

with st.spinner("Scanning Matrix..."):
    try:
        data = yf.download(watchlist, period=lookback, session=session, group_by='ticker', progress=False)
        if data.empty:
            st.error("Provider returned no data.")
        else:
            valid_ts = data.columns.get_level_values(0).unique() if isinstance(data.columns, pd.MultiIndex) else [watchlist]
            for t in watchlist:
                if t not in valid_ts: continue
                try:
                    h = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                    h = h.dropna(subset=['Close'])
                    h = calculate_indicators(h)
                    if h is not None:
                        processed.append({
                            "Asset": t, "Score": get_score(h), "Price": round(h.iloc[-1]['Close'], 4),
                            "RSI": round(h.iloc[-1]['RSI'], 1), "ADX": round(h.iloc[-1]['ADX'], 1),
                            "Status": "🟢 Bull" if h.iloc[-1]['Trend'] == 1 else "🔴 Bear"
                        })
                except: continue
    except Exception as e:
        st.error(f"Download Error: {e}")

# 5. UI DISPLAY
if processed:
    df_final = pd.DataFrame(processed).sort_values("Score", ascending=False)
    st.dataframe(df_final, width="stretch", hide_index=True)
    
    st.markdown("---")
    st.subheader("🎯 High-Conviction Alerts")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Validated Breakouts**")
        for _, r in df_final[(df_final['Score'] == 4) & (df_final['ADX'] >= 25)].iterrows():
            st.success(f"{r['Asset']} | ADX: {r['ADX']}")
            
    with c2:
        st.write("**Range Traps**")
        for _, r in df_final[(df_final['Score'] == 4) & (df_final['ADX'] < 15)].iterrows():
            st.warning(f"{r['Asset']} | ADX: {r['ADX']}")
else:
    st.info("No assets parsed. Check engine logs.")

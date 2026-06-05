import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime

st.set_page_config(page_title="Alpha Engine", layout="wide", initial_sidebar_state="expanded")
st.title("🛡️ Alpha Engine")

st.markdown("""<style>.stMetric {background-color: rgba(128,128,128,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);}</style>""", unsafe_allow_html=True)

default_tickers = ["MRVL", "PLUG", "RGTI", "IREN", "QBTS", "CRWV", "MSTR", "CIFR", "IONQ", "HOOD", "CLSK", "WULF", "QUBT", "CORZ", "NBIS", "BE", "TSM", "CBRS", "HUT", "TTWO", "CAT", "MU", "META", "TSLA", "AVGO", "MSFT", "GOOGL", "AAPL", "AMZN", "HON", "NVDA", "ORCL", "CRM", "PLTR", "VVV", "MET", "RARE", "BTC-USD", "ETH-USD", "BNB-USD", "ZEC-USD", "XMR-USD", "SOL-USD", "QNT-USD", "LTC-USD", "DASH-USD", "LINK-USD", "INJ-USD", "ICP-USD", "NEAR-USD", "XRP-USD", "XLM-USD", "DOGE-USD", "JUP-USD", "WIF-USD", "BONK-USD", "SHIB-USD", "GC=F", "SI=F", "PL=F", "PA=F", "CL=F", "BZ=F", "HG=F", "LIT", "GLTR", "PALL", "REMX", "SIL", "BOTZ", "IGV", "AIQU", "REXC"]
if "persisted_tickers" not in st.session_state: st.session_state["persisted_tickers"] = ", ".join(default_tickers)

st.sidebar.header("🎛️ Control Panel")
if st.sidebar.button("🔄 Force Hard Reload", use_container_width=True): st.cache_data.clear()
score_selection = st.sidebar.slider("Filter Confluence Score", min_value=-6, max_value=6, value=(-6, 6), step=1)
min_score, max_score = score_selection if isinstance(score_selection, tuple) else (score_selection, score_selection)

with st.sidebar.expander("📝 Edit Assets", expanded=False):
    t_input = st.sidebar.text_area("Tickers", value=st.session_state["persisted_tickers"], key="ticker_input_field", height=250)
    st.session_state["persisted_tickers"] = t_input
watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]
lookback = st.sidebar.selectbox("Lookback", ["6mo", "3mo", "1y"], index=0)

def calc(df):
    if len(df) < 55: return None
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + (gain.ewm(com=13, adjust=False).mean() / (loss.ewm(com=13, adjust=False).mean() + 1e-10))))
    e12, e26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    up, down = df['High'].diff(), -df['Low'].diff()
    df['+DM'] = np.where((up > down) & (up > 0), up, 0)
    df['-DM'] = np.where((down > up) & (down > 0), down, 0)
    tr_s = df['TR'].ewm(com=13, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(com=13, adjust=False).mean() / (tr_s + 1e-10))
    df['-DI'] = 100 * (df['-DM'].ewm(com=13, adjust=False).mean() / (tr_s + 1e-10))
    df['ADX'] = (100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-10)).ewm(com=13, adjust=False).mean()
    return df

@st.cache_data(ttl=600)
def get_data(tickers, period):
    data = yf.download(tickers, period=period, group_by='ticker', progress=False, timeout=15)
    res, errs = [], []
    valid = data.columns.get_level_values(0).unique() if isinstance(data.columns, pd.MultiIndex) else ([tickers] if 'Close' in data.columns else [])
    for t in tickers:
        if t not in valid:
            errs.append({"Asset": t, "Reason": "Omitted"})
            continue
        try:
            h = calc(data[t].copy().dropna(subset=['Close']) if isinstance(data.columns, pd.MultiIndex) else data.copy().dropna(subset=['Close']))
            if h is not None:
                r, p = h.iloc[-1], h.iloc[-2]
                score = sum([1 if r['Close'] > r['EMA5'] else -1, 1 if r['Close'] > r['EMA20'] else -1, 1 if r['Close'] > r['EMA50'] else -1, 1 if r['RSI'] > 50 else -1, 1 if r['MACD_Hist'] > 0 else -1, 1 if r['+DI'] > r['-DI'] else -1])
                status = "⚪ Neutral" if r['ADX'] < 15 else ("🚀 Bullish Flip" if r['Close'] > r['EMA20'] and p['Close'] <= p['EMA20'] else ("🩸 Bearish Flip" if r['Close'] < r['EMA20'] and p['Close'] >= p['EMA20'] else ("🟢 Bullish" if r['Close'] > r['EMA20'] else "🔴 Bearish")))
                res.append({"Asset": t, "Score": score, "Status": status, "Price": round(r['Close'], 4), "RSI": round(r['RSI'], 1), "ADX": round(r['ADX'], 1)})
        except Exception as e: errs.append({"Asset": t, "Reason": str(e)})
    return res, errs, datetime.datetime.now()

proc, failed, last_up = get_data(tuple(watchlist), lookback)
st.sidebar.markdown(f"⏱️ *Parsed at:* `{last_up.strftime('%H:%M:%S')}`")

if proc:
    df = pd.DataFrame(proc)
    df_f = df[(df['Score'] >= min_score) & (df['Score'] <= max_score)].sort_values("Score", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df)); c2.metric("Filtered", len(df_f)); c3.metric("Bullish 🚀", len(df[df['Status'] == "🚀 Bullish Flip"])); c4.metric("Bearish 🩸", len(df[df['Status'] == "🩸 Bearish Flip"]))
    st.markdown("---")
    t1, t2, t3 = st.tabs(["📊 Matrix", "🎯 Alerts", "🔍 Logs"])
    with t1: st.dataframe(df_f[["Asset", "Score", "Status", "Price", "RSI", "ADX"]], width="stretch", hide_index=True)
    with t2:
        l, r = st.columns(2)
        l.markdown("#### 🔥 Structural Momentum"); [l.success(f"{row['Asset']} | {row['Price']} | ADX:{row['ADX']}") for _, row in df_f[(df_f['Score']==6)&(df_f['ADX']>=25)].iterrows()] or l.info("None")
        r.markdown("#### 💤 Range Traps"); [r.warning(f"{row['Asset']} | {row['Price']} | ADX:{row['ADX']}") for _, row in df_f[(df_f['Score']==6)&(df_f['ADX']<15)].iterrows()] or r.info("None")
    with t3:
        if failed:
            st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True)
        else:
            st.success("All systems green.")
else: st.error("Processing...")

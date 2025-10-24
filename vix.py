# streamlit_vix_sp500_tsla_signal_fixed.py
# 修正版：更健壮的 fetch_data，已修复之前的 SyntaxError 并改进数据校验
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

st.set_page_config(page_title="VIX & S&P500 → TSLA Signal", layout="wide")

st.title("实时监控：VIX、S&P500 与 TSLA 买/卖 建议（修正版）")
st.markdown("示例策略，仅供学习。")

# --------------------
# User controls
# --------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    refresh_seconds = st.number_input("自动刷新秒数（0 关闭）", min_value=0, value=0, step=10)
with col2:
    history_days = st.number_input("用于计算均线/指标的历史天数", min_value=30, max_value=3650, value=180, step=30)
with col3:
    vix_threshold_sell = st.number_input("VIX 卖出阈值", min_value=5.0, value=25.0, step=0.5)

positionsize = st.slider("建议最大仓位占净值比例（模拟）", min_value=1, max_value=100, value=10)

if refresh_seconds > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        # call it; it will rerun app automatically
        count = st_autorefresh(interval=refresh_seconds * 1000, limit=None)
    except Exception:
        st.info("自动刷新需要 'streamlit-autorefresh' 包；已禁用自动刷新，请手动刷新页面。")

# --------------------
# Robust data fetcher
# --------------------
@st.cache_data(ttl=30)
def fetch_data(ticker: str, period_days: int = 180):
    # request extra days for moving averages
    period_str = f"{max(period_days + 30, 60)}d"
    data = yf.download(ticker, period=period_str, interval='1d', progress=False)
    if data is None or data.empty:
        return pd.DataFrame()

    # Normalize column names: prefer 'Adj Close' if present, else 'Close'.
    cols = [c.lower() for c in data.columns]

    # Try to map possible column names to canonical names
    # We'll create a DataFrame with at least 'Adj Close' or 'Close' as 'Adj Close'
    df = data.copy()
    # If 'Adj Close' exists, keep it; else if only 'Close' exists, copy to 'Adj Close'
    if 'Adj Close' in df.columns:
        pass
    elif 'Close' in df.columns:
        df['Adj Close'] = df['Close']
    else:
        # Unlikely, but bail out
        return pd.DataFrame()

    # Ensure we have numeric index of datetimes
    df.index = pd.to_datetime(df.index)

    # Keep only the columns we need if they exist
    wanted = ['Open','High','Low','Close','Adj Close','Volume']
    kept = [c for c in wanted if c in df.columns]
    df = df[kept]

    # Some tickers (like ^VIX) may not have Volume/Open/High/Low - that's fine
    return df

# --------------------
# Fetch data
# --------------------
with st.spinner("拉取行情中..."):
    vix_df = fetch_data('^VIX', period_days=int(history_days))
    spx_df = fetch_data('^GSPC', period_days=int(history_days))
    tsla_df = fetch_data('TSLA', period_days=int(history_days))

last_update = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
st.sidebar.metric("最后更新时间", last_update)

if vix_df.empty or spx_df.empty or tsla_df.empty:
    st.error("部分数据拉取失败（可能 yfinance 在当前环境受限或标的返回列结构不同）。请检查网络或尝试增加 period_days。")
    st.stop()

# --------------------
# Indicators
# --------------------
def add_indicators(df):
    df = df.copy()
    # ensure Adj Close exists
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        else:
            df['Adj Close'] = np.nan
    df['ret'] = df['Adj Close'].pct_change()
    df['ma5'] = df['Adj Close'].rolling(5).mean()
    df['ma20'] = df['Adj Close'].rolling(20).mean()
    df['ma50'] = df['Adj Close'].rolling(50).mean()
    df['vol10'] = df['ret'].rolling(10).std() * np.sqrt(252)
    return df

vix_df = add_indicators(vix_df)
spx_df = add_indicators(spx_df)
tsla_df = add_indicators(tsla_df)

# Recent values (safely)
vix_now = float(vix_df['Adj Close'].iloc[-1])
spx_now = float(spx_df['Adj Close'].iloc[-1])
tsla_now = float(tsla_df['Adj Close'].iloc[-1])

vix_5d_change = (vix_df['Adj Close'].iloc[-1] / vix_df['Adj Close'].iloc[-6] - 1) if len(vix_df) >= 6 else np.nan
spx_ma5 = spx_df['ma5'].iloc[-1] if not np.isnan(spx_df['ma5'].iloc[-1]) else spx_df['Adj Close'].iloc[-1]
spx_ma20 = spx_df['ma20'].iloc[-1] if not np.isnan(spx_df['ma20'].iloc[-1]) else spx_df['Adj Close'].rolling(20).mean().iloc[-1]
tsla_ma20 = tsla_df['ma20'].iloc[-1]

# --------------------
# Signal logic
# --------------------
signal = 'HOLD'
confidence = 'Low'
reason = []

if (vix_now >= vix_threshold_sell) or (not np.isnan(vix_5d_change) and vix_5d_change > 0.15 and spx_now < spx_ma20):
    signal = 'STRONG SELL'
    confidence = 'High'
    reason.append(f"VIX={vix_now:.2f} >= {vix_threshold_sell}")
elif (vix_now > 20) and (spx_ma5 < spx_ma20):
    signal = 'SELL'
    confidence = 'Medium'
    reason.append(f"VIX={vix_now:.2f} > 20 且 SPX MA5 < MA20")
elif (vix_now < 15) and (spx_ma5 > spx_ma20) and (tsla_now > tsla_ma20):
    signal = 'BUY'
    confidence = 'Medium'
    reason.append(f"VIX={vix_now:.2f} < 15 且 SPX 趋势向上 且 TSLA > MA20")
else:
    signal = 'HOLD'
    confidence = 'Low'
    reason.append('未满足明确买卖条件')

if signal == 'BUY':
    suggested_size = f"可建仓约 {positionsize}% 的净值（模拟），建议分批进场"
elif signal in ['SELL','STRONG SELL']:
    suggested_size = "建议减少或清仓；如需防守可考虑买入保护性看跌期权"
else:
    suggested_size = "建议观望"

# --------------------
# UI
# --------------------
colA, colB, colC, colD = st.columns(4)
colA.metric("VIX", f"{vix_now:.2f}", delta=f"{vix_df['Adj Close'].pct_change().iloc[-1]*100:.2f}%")
colB.metric("S&P500", f"{spx_now:.2f}", delta=f"{spx_df['Adj Close'].pct_change().iloc[-1]*100:.2f}%")
colC.metric("TSLA", f"{tsla_now:.2f}", delta=f"{tsla_df['Adj Close'].pct_change().iloc[-1]*100:.2f}%")
colD.metric("建议信号", signal, delta=f"信心水平: {confidence}")

st.markdown("### 生成信号的理由")
st.write("; ".join(reason))
st.markdown("**仓位建议**")
st.write(suggested_size)

# --------------------
# Charts
# --------------------
lookback = int(min(len(tsla_df), history_days))

chart_col1, chart_col2 = st.columns([2,1])
with chart_col1:
    st.subheader('价格与均线：TSLA 与 S&P500')
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(tsla_df.index[-lookback:], tsla_df['Adj Close'].iloc[-lookback:], label='TSLA')
    ax[0].plot(tsla_df.index[-lookback:], tsla_df['ma20'].iloc[-lookback:], label='TSLA MA20')
    ax[0].set_ylabel('TSLA'); ax[0].legend()
    ax[1].plot(spx_df.index[-lookback:], spx_df['Adj Close'].iloc[-lookback:], label='S&P500')
    ax[1].plot(spx_df.index[-lookback:], spx_df['ma20'].iloc[-lookback:], label='SPX MA20')
    ax[1].set_ylabel('SPX'); ax[1].legend()
    st.pyplot(fig)

with chart_col2:
    st.subheader('VIX 与 历史波动')
    fig2, ax2 = plt.subplots(2,1, figsize=(5,6), sharex=True)
    ax2[0].plot(vix_df.index[-lookback:], vix_df['Adj Close'].iloc[-lookback:], label='VIX')
    ax2[0].axhline(15, linestyle='--'); ax2[0].axhline(20, linestyle=':')
    ax2[0].set_ylabel('VIX'); ax2[0].legend()
    ax2[1].plot(tsla_df.index[-lookback:], tsla_df['vol10'].iloc[-lookback:], label='TSLA vol (10d)')
    ax2[1].set_ylabel('Vol'); ax2[1].legend()
    st.pyplot(fig2)

# --------------------
# Data & download
# --------------------
st.markdown('---')
st.subheader('原始数据（最近若干行）')
display_df = pd.concat([
    vix_df['Adj Close'].rename('VIX').tail(lookback),
    spx_df['Adj Close'].rename('SPX').tail(lookback),
    tsla_df['Adj Close'].rename('TSLA').tail(lookback)
], axis=1)
st.dataframe(display_df)

csv = display_df
st.download_button('下载CSV（最近数据）', csv.tail(lookback).to_csv().encode('utf-8'), file_name='vix_spx_tsla.csv')

st.markdown('---')
st.caption('策略示例：仅演示如何结合 VIX 与 SPX 给出 TSLA 的简易买卖建议。真实交易请回测并设置风控。')

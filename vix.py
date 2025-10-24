# streamlit_vix_sp500_tsla_signal.py
# Streamlit app: Real-time monitor of VIX, S&P500 and TSLA -> issue buy/sell/hold suggestions
# Requirements:
#   pip install streamlit yfinance pandas numpy matplotlib
# Optional for auto refresh:
#   pip install streamlit-autorefresh

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import time

st.set_page_config(page_title="VIX & S&P500 → TSLA Signal", layout="wide")

st.title("实时监控：VIX、S&P500 与 TSLA 买/卖 建议")
st.markdown("""
此应用定期拉取 VIX (^VIX)、S&P500 (^GSPC) 与 TSLA (^TSLA) 的最新行情与历史数据，
并基于简单量化规则给出**买入 / 持有 / 卖出** 建议。

**提示**：此策略为教学用途，不是投资建议。请在实盘前自行回测并设置风险管理。
""")

# --------------------
# User controls
# --------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    refresh_seconds = st.number_input("自动刷新秒数（0 表示关闭自动刷新）", min_value=0, value=0, step=10)
with col2:
    history_days = st.number_input("用于计算均线/指标的历史天数", min_value=30, max_value=3650, value=180, step=30)
with col3:
    vix_threshold_sell = st.number_input("VIX 卖出阈值", min_value=5.0, value=25.0, step=0.5)

positionsize = st.slider("建议最大仓位占净值比例（模拟）", min_value=1, max_value=100, value=10)

# Auto-refresh: try to use streamlit-autorefresh if installed
if refresh_seconds > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        # call it; it will rerun app automatically
        count = st_autorefresh(interval=refresh_seconds * 1000, limit=None)
    except Exception:
        # fallback: show a note and provide manual refresh
        st.info("自动刷新需要 'streamlit-autorefresh' 包。已禁用自动刷新；请手动点击右上角的刷新或使用“刷新”按钮。")

# --------------------
# Data fetching
# --------------------
@st.cache_data(ttl=30)
def fetch_data(ticker, period_days=history_days):
    # yfinance period uses strings; we'll request a bit extra for moving averages
    period_str = f"{max(period_days + 30, 60)}d"
    data = yf.download(ticker, period=period_str, interval='1d', progress=False)
    if data.empty:
        return pd.DataFrame()
    data = data[['Open','High','Low','Close','Adj Close','Volume']]
    data.index = pd.to_datetime(data.index)
    return data

with st.spinner("拉取行情中..."):
    vix_df = fetch_data('^VIX')
    spx_df = fetch_data('^GSPC')
    tsla_df = fetch_data('TSLA')

last_update = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
st.sidebar.metric("最后更新时间", last_update)

# Validate
if vix_df.empty or spx_df.empty or tsla_df.empty:
    st.error("数据拉取失败。请检查网络或 yfinance 是否可用。")
    st.stop()

# --------------------
# Feature engineering
# --------------------
lookback = int(min(len(tsla_df), history_days))

def add_indicators(df):
    df = df.copy()
    df['ret'] = df['Adj Close'].pct_change()
    df['ma5'] = df['Adj Close'].rolling(5).mean()
    df['ma20'] = df['Adj Close'].rolling(20).mean()
    df['ma50'] = df['Adj Close'].rolling(50).mean()
    df['vol10'] = df['ret'].rolling(10).std() * np.sqrt(252)
    return df

vix_df = add_indicators(vix_df)
spx_df = add_indicators(spx_df)
tsla_df = add_indicators(tsla_df)

# Recent values
vix_now = vix_df['Adj Close'].iloc[-1]
spx_now = spx_df['Adj Close'].iloc[-1]
tsla_now = tsla_df['Adj Close'].iloc[-1]

vix_5d_change = (vix_df['Adj Close'].iloc[-1] / vix_df['Adj Close'].iloc[-6] - 1) if len(vix_df) >= 6 else np.nan
spx_ma5 = spx_df['ma5'].iloc[-1]
spx_ma20 = spx_df['ma20'].iloc[-1]

# TSLA momentum
tsla_ma20 = tsla_df['ma20'].iloc[-1]

# --------------------
# Signal logic (simple, interpretable rules)
# --------------------
# Rules (example):
# 1) Strong Sell: VIX > vix_threshold_sell OR (VIX 5d change > +0.15 AND SPX below MA20)
# 2) Sell: VIX > 20 and SPX MA5 < MA20
# 3) Buy: VIX < 15 and SPX MA5 > MA20 and TSLA price > MA20
# 4) Hold: otherwise

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
    reason.append(f"VIX={vix_now:.2f} > 20 and SPX MA5 < MA20")
elif (vix_now < 15) and (spx_ma5 > spx_ma20) and (tsla_now > tsla_ma20):
    signal = 'BUY'
    confidence = 'Medium'
    reason.append(f"VIX={vix_now:.2f} < 15 and SPX 趋势向上 and TSLA > MA20")
else:
    signal = 'HOLD'
    confidence = 'Low'
    reason.append('未满足明确买卖条件')

# Position sizing suggestion
if signal == 'BUY':
    suggested_size = f"可建仓约 {positionsize}% 的净值（模拟）——分批进场，首次仓位建议 {max(1,int(positionsize/2))}%" 
elif signal in ['SELL','STRONG SELL']:
    suggested_size = "建议减少或清仓；如有空头权限，可择机建空或买入保护性看跌期权。"
else:
    suggested_size = "建议观望，等待更明确的信号或用期权对冲当前持仓风险。"

# --------------------
# UI: Top summary
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
# Plots
# --------------------
st.markdown("---")
chart_col1, chart_col2 = st.columns([2,1])
with chart_col1:
    st.subheader('价格与均线：TSLA 与 S&P500')
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(tsla_df.index[-lookback:], tsla_df['Adj Close'].iloc[-lookback:], label='TSLA')
    ax[0].plot(tsla_df.index[-lookback:], tsla_df['ma20'].iloc[-lookback:], label='TSLA MA20')
    ax[0].set_ylabel('TSLA')
    ax[0].legend()

    ax[1].plot(spx_df.index[-lookback:], spx_df['Adj Close'].iloc[-lookback:], label='S&P500')
    ax[1].plot(spx_df.index[-lookback:], spx_df['ma20'].iloc[-lookback:], label='SPX MA20')
    ax[1].set_ylabel('SPX')
    ax[1].legend()
    st.pyplot(fig)

with chart_col2:
    st.subheader('VIX 与 历史波动')
    fig2, ax2 = plt.subplots(2,1, figsize=(5,6), sharex=True)
    ax2[0].plot(vix_df.index[-lookback:], vix_df['Adj Close'].iloc[-lookback:], label='VIX')
    ax2[0].axhline(15, linestyle='--')
    ax2[0].axhline(20, linestyle=':')
    ax2[0].set_ylabel('VIX')
    ax2[0].legend()

    ax2[1].plot(tsla_df.index[-lookback:], tsla_df['vol10'].iloc[-lookback:], label='TSLA annualized vol (10d)')
    ax2[1].set_ylabel('Vol')
    ax2[1].legend()
    st.pyplot(fig2)

# --------------------
# Details & downloads
# --------------------
st.markdown('---')
st.subheader('原始数据（最后 %d 行）' % min(50, lookback))
st.dataframe(pd.concat([vix_df[['Adj Close']].rename(columns={'Adj Close':'VIX'}).tail(lookback), 
                         spx_df[['Adj Close']].rename(columns={'Adj Close':'SPX'}).tail(lookback),
                         tsla_df[['Adj Close']].rename(columns={'Adj Close':'TSLA'}).tail(lookback)], axis=1))

csv = pd.concat([vix_df['Adj Close'].rename('VIX'), spx_df['Adj Close'].rename('SPX'), tsla_df['Adj Close'].rename('TSLA')], axis=1)

st.download_button('下载CSV（最近数据）', csv.tail(lookback).to_csv().encode('utf-8'), file_name='vix_spx_tsla.csv')

# --------------------
# Footnote / Risk
# --------------------
st.markdown('---')
st.caption('策略说明：规则基于简单阈值与短期均线交叉，作为示例展示。真实交易应加入资金管理、滑点、委托策略与回测验证。')

# Manual refresh button
if st.button('手动刷新'):
    st.experimental_rerun()

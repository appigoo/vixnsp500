import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# 页面标题
st.title("VIX & SP500 实时监控与 TSLA 买卖建议")

# 用户输入部分：API 或直接使用 yfinance（无需密钥）
st.sidebar.header("设置")
refresh_interval = st.sidebar.slider("刷新间隔（秒）", 10, 300, 60)

# 策略参数
st.sidebar.header("买卖策略参数")
vix_threshold_high = st.sidebar.slider("VIX 高阈值（恐慌卖出）", 15.0, 50.0, 30.0)
vix_threshold_low = st.sidebar.slider("VIX 低阈值（安全买入）", 10.0, 25.0, 15.0)
sp500_trend_days = st.sidebar.slider("SP500 趋势天数", 5, 20, 10)

# 简单策略说明
st.sidebar.markdown("""
### 策略逻辑
- **买入建议**：VIX < 低阈值 且 SP500 过去 N 天上涨 > 2%。
- **卖出建议**：VIX > 高阈值 或 SP500 过去 N 天下跌 > 2%。
- **持有**：其他情况。
""")

# 函数：获取实时数据
@st.cache_data(ttl=refresh_interval)
def fetch_data():
    # 获取 VIX (^VIX)
    vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
    current_vix = vix['Close'].iloc[-1] if not vix.empty else None
    
    # 获取 SP500 (^GSPC)
    sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days + 1}d", interval="1d")
    current_sp500 = sp500['Close'].iloc[-1] if not sp500.empty else None
    sp500_trend = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[0]) / sp500['Close'].iloc[0]) * 100
    
    # 获取 TSLA 当前价（用于参考）
    tsla = yf.Ticker("TSLA").history(period="1d", interval="1m")
    current_tsla = tsla['Close'].iloc[-1] if not tsla.empty else None
    
    return {
        'vix': current_vix,
        'sp500': current_sp500,
        'sp500_trend': sp500_trend,
        'tsla': current_tsla,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# 主循环：实时更新
placeholder = st.empty()

while True:
    data = fetch_data()
    
    with placeholder.container():
        # 显示当前时间
        st.metric("更新时间", data['timestamp'])
        
        # 显示指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VIX 指数", f"{data['vix']:.2f}" if data['vix'] else "N/A")
        with col2:
            st.metric("SP500 指数", f"{data['sp500']:.2f}" if data['sp500'] else "N/A")
        with col3:
            st.metric("TSLA 股价", f"{data['tsla']:.2f}" if data['tsla'] else "N/A")
        
        # SP500 趋势
        st.metric("SP500 趋势 (%)", f"{data['sp500_trend']:.2f}%")
        
        # 买卖建议
        suggestion = "持有"
        color = "off"
        
        if data['vix'] > vix_threshold_high:
            suggestion = "🚨 卖出 TSLA"
            color = "inverse"
        elif data['vix'] < vix_threshold_low and data['sp500_trend'] > 2:
            suggestion = "💰 买入 TSLA"
            color = "normal"
        elif data['sp500_trend'] < -2:
            suggestion = "⚠️ 卖出 TSLA"
            color = "inverse"
        
        st.error(suggestion) if "卖出" in suggestion else st.success(suggestion) if "买入" in suggestion else st.info(suggestion)
        
        # 数据表格（最近趋势）
        if 'sp500' in data:
            recent_sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days}d")
            st.subheader("SP500 最近趋势")
            st.dataframe(recent_sp500.tail(5), use_container_width=True)
    
    time.sleep(refresh_interval)
    st.rerun()

# 运行说明
st.markdown("---")
st.markdown("""
### 运行说明
1. 安装依赖：`pip install streamlit yfinance pandas`
2. 运行程序：`streamlit run this_script.py`
3. 程序将每 X 秒自动刷新数据（yfinance 提供近实时数据，延迟约 1-5 分钟）。
4. **注意**：这仅为教育性示例，非投资建议。实际交易需谨慎，考虑风险。
""")

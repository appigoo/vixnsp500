import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

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

# 预测参数
st.sidebar.header("VIX 预测参数")
arima_order = (1, 1, 1)  # ARIMA(p,d,q) 参数，可调整

# 简单策略说明
st.sidebar.markdown("""
### 策略逻辑
- **买入建议**：VIX < 低阈值 且 SP500 过去 N 天上涨 > 2%。
- **卖出建议**：VIX > 高阈值 或 SP500 过去 N 天下跌 > 2%。
- **持有**：其他情况。
""")

# 函数：获取实时数据
@st.cache_data(ttl=refresh_interval)
def fetch_data(sp500_trend_days):
    # 获取 VIX (^VIX) - 分钟数据用于图表和预测
    vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
    if not vix.empty:
        current_vix = vix['Close'].iloc[-1]
        vix_df = vix[['Close']].copy()
        vix_df.columns = ['VIX']
    else:
        current_vix = None
        vix_df = pd.DataFrame()
    
    # 获取 SP500 (^GSPC) - 分钟数据用于图表
    sp500_min = yf.Ticker("^GSPC").history(period="1d", interval="1m")
    if not sp500_min.empty:
        sp500_df = sp500_min[['Close']].copy()
        sp500_df.columns = ['SP500']
    else:
        sp500_df = pd.DataFrame()
    
    # 获取 SP500 日数据用于趋势计算
    sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days + 1}d", interval="1d")
    if not sp500.empty:
        current_sp500 = sp500['Close'].iloc[-1]
        sp500_trend = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[0]) / sp500['Close'].iloc[0]) * 100
    else:
        current_sp500 = None
        sp500_trend = 0.0
    
    # 获取 TSLA 当前价（用于参考）
    tsla = yf.Ticker("TSLA").history(period="1d", interval="1m")
    current_tsla = tsla['Close'].iloc[-1] if not tsla.empty else None
    
    return {
        'vix': current_vix,
        'vix_df': vix_df,
        'sp500': current_sp500,
        'sp500_df': sp500_df,
        'sp500_trend': sp500_trend,
        'tsla': current_tsla,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# 函数：VIX 下一分钟预测
def predict_next_vix(vix_df, order=arima_order):
    if len(vix_df) < 10:  # 需要足够数据点
        return None, "数据不足，无法预测"
    
    try:
        # 使用 ARIMA 模型预测
        model = ARIMA(vix_df['VIX'], order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        next_vix = forecast.iloc[0]
        confidence_interval = model_fit.get_forecast(steps=1).conf_int().iloc[0].to_dict()
        
        return next_vix, f"预测值: {next_vix:.2f} (95% CI: {confidence_interval['lower VIX']:.2f} - {confidence_interval['upper VIX']:.2f})"
    except Exception as e:
        return None, f"预测错误: {str(e)}"

# 主循环：实时更新
placeholder = st.empty()

while True:
    data = fetch_data(sp500_trend_days)
    
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
        
        # VIX 实时走势图
        if not data['vix_df'].empty:
            st.subheader("VIX 实时走势图 (最近1天分钟数据)")
            st.line_chart(data['vix_df'])
        
        # SP500 实时走势图
        if not data['sp500_df'].empty:
            st.subheader("SP500 实时走势图 (最近1天分钟数据)")
            st.line_chart(data['sp500_df'])
        
        # VIX 下一分钟预测
        next_vix, pred_msg = predict_next_vix(data['vix_df'])
        if next_vix is not None:
            delta = next_vix - data['vix']
            trend = "上涨" if delta > 0 else "下跌" if delta < 0 else "持平"
            st.metric(f"下一分钟 VIX 预测 ({trend})", f"{next_vix:.2f}", f"{delta:+.2f}")
        else:
            st.warning(pred_msg)
        
        # 买卖建议
        suggestion = "持有"
        
        if data['vix'] is not None and data['vix'] > vix_threshold_high:
            suggestion = "🚨 卖出 TSLA"
        elif data['vix'] is not None and data['vix'] < vix_threshold_low and data['sp500_trend'] > 2:
            suggestion = "💰 买入 TSLA"
        elif data['sp500_trend'] < -2:
            suggestion = "⚠️ 卖出 TSLA"
        
        # 使用 if-elif 显示建议
        if "卖出" in suggestion:
            st.error(suggestion)
        elif "买入" in suggestion:
            st.success(suggestion)
        else:
            st.info(suggestion)
        
        # 数据表格（最近趋势）
        recent_sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days}d")
        st.subheader("SP500 最近趋势")
        st.dataframe(recent_sp500.tail(5), width='stretch')
    
    time.sleep(refresh_interval)
    st.rerun()

# 运行说明
st.markdown("---")
st.markdown("""
### 运行说明
1. 安装依赖：`pip install streamlit yfinance pandas numpy statsmodels`
2. 运行程序：`streamlit run this_script.py`
3. 程序将每 X 秒自动刷新数据（yfinance 提供近实时数据，延迟约 1-5 分钟）。
4. **VIX 预测**：使用简单 ARIMA 模型基于最近分钟数据预测下一分钟值，仅供参考，非准确预测。
5. **注意**：这仅为教育性示例，非投资建议。实际交易需谨慎，考虑风险。
""")

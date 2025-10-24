import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

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
enable_grid_search_vix = st.sidebar.checkbox("启用动态参数优化 (VIX ARIMA网格搜索)", value=True)
p_max_vix = st.sidebar.slider("VIX ARIMA p 最大值", 0, 5, 2)
d_max_vix = st.sidebar.slider("VIX ARIMA d 最大值", 0, 2, 1)
q_max_vix = st.sidebar.slider("VIX ARIMA q 最大值", 0, 5, 2)

# TSLA ARIMA 预测参数
st.sidebar.header("TSLA ARIMA 预测参数")
enable_grid_search_tsla = st.sidebar.checkbox("启用动态参数优化 (TSLA ARIMA网格搜索)", value=True)
p_max_tsla = st.sidebar.slider("TSLA ARIMA p 最大值", 0, 5, 2)
d_max_tsla = st.sidebar.slider("TSLA ARIMA d 最大值", 0, 2, 1)
q_max_tsla = st.sidebar.slider("TSLA ARIMA q 最大值", 0, 5, 2)
tsla_forecast_steps = st.sidebar.slider("TSLA 多步预测步数 (分钟)", 1, 10, 5)

# TSLA 相关性预测参数
st.sidebar.header("TSLA 相关性预测参数")
enable_tsla_corr_predict = st.sidebar.checkbox("启用基于VIX-TSLA相关性的TSLA预测", value=True)
corr_window = st.sidebar.slider("相关性计算窗口 (分钟)", 5, 30, 5)

# 图表优化参数
st.sidebar.header("图表优化")
ma_window = st.sidebar.slider("移动平均窗口期 (用于趋势线)", 3, 20, 5)

# 简单策略说明
st.sidebar.markdown("""
### 策略逻辑
- **买入建议**：VIX < 低阈值 且 SP500 过去 N 天上涨 > 2%。
- **卖出建议**：VIX > 高阈值 或 SP500 过去 N 天下跌 > 2%。
- **持有**：其他情况。
""")

# 函数：获取实时数据
@st.cache_data(ttl=refresh_interval)
def fetch_data(sp500_trend_days, ma_window, corr_window):
    # 获取 VIX (^VIX) - 分钟数据用于图表和预测
    vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
    if not vix.empty:
        current_vix = vix['Close'].iloc[-1]
        vix_open = vix['Open'].iloc[0]
        vix_change_pct = ((current_vix - vix_open) / vix_open) * 100 if vix_open != 0 else 0
        vix_df = vix[['Close']].copy()
        vix_df.columns = ['VIX']
        vix_df['VIX_MA'] = vix_df['VIX'].rolling(window=ma_window, min_periods=1).mean()
    else:
        current_vix = None
        vix_change_pct = 0
        vix_df = pd.DataFrame()
    
    # 获取 SP500 (^GSPC) - 分钟数据用于图表
    sp500_min = yf.Ticker("^GSPC").history(period="1d", interval="1m")
    if not sp500_min.empty:
        current_sp500_min = sp500_min['Close'].iloc[-1]
        sp500_open = sp500_min['Open'].iloc[0]
        sp500_change_pct = ((current_sp500_min - sp500_open) / sp500_open) * 100 if sp500_open != 0 else 0
        sp500_df = sp500_min[['Close']].copy()
        sp500_df.columns = ['SP500']
        sp500_df['SP500_MA'] = sp500_df['SP500'].rolling(window=ma_window, min_periods=1).mean()
    else:
        current_sp500_min = None
        sp500_change_pct = 0
        sp500_df = pd.DataFrame()
    
    # 获取 SP500 日数据用于趋势计算
    sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days + 1}d", interval="1d")
    if not sp500.empty:
        current_sp500 = sp500['Close'].iloc[-1]
        sp500_trend = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[0]) / sp500['Close'].iloc[0]) * 100
    else:
        current_sp500 = None
        sp500_trend = 0.0
    
    # 获取 TSLA - 分钟数据用于图表和当前价
    tsla = yf.Ticker("TSLA").history(period="1d", interval="1m")
    if not tsla.empty:
        current_tsla = tsla['Close'].iloc[-1]
        tsla_open = tsla['Open'].iloc[0]
        tsla_change_pct = ((current_tsla - tsla_open) / tsla_open) * 100 if tsla_open != 0 else 0
        tsla_df = tsla[['Close']].copy()
        tsla_df.columns = ['TSLA']
        tsla_df['TSLA_MA'] = tsla_df['TSLA'].rolling(window=ma_window, min_periods=1).mean()
    else:
        current_tsla = None
        tsla_change_pct = 0
        tsla_df = pd.DataFrame()
    
    return {
        'vix': current_vix,
        'vix_df': vix_df,
        'vix_change_pct': vix_change_pct,
        'sp500': current_sp500,
        'sp500_df': sp500_df,
        'sp500_change_pct': sp500_change_pct,
        'sp500_trend': sp500_trend,
        'tsla': current_tsla,
        'tsla_df': tsla_df,
        'tsla_change_pct': tsla_change_pct,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# 函数：VIX 下一分钟预测（优化版：动态网格搜索最佳参数）
def predict_next_vix(vix_df, enable_grid=True, p_max=2, d_max=1, q_max=2):
    if len(vix_df) < 20:  # 需要足够数据点
        return None, "数据不足，无法预测"
    
    vix_series = vix_df['VIX'].dropna()
    
    if enable_grid:
        # 动态网格搜索最佳ARIMA参数
        p_range = range(0, p_max + 1)
        d_range = range(0, d_max + 1)
        q_range = range(0, q_max + 1)
        param_combinations = list(product(p_range, d_range, q_range))
        
        best_aic = float("inf")
        best_order = None
        best_forecast = None
        
        for order in param_combinations:
            try:
                model = ARIMA(vix_series, order=order)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = order
                    forecast = fitted_model.forecast(steps=1)
                    best_forecast = forecast.iloc[0]
            except:
                continue
        
        if best_forecast is not None:
            return best_forecast, f"最佳参数: ARIMA{best_order} (AIC: {best_aic:.2f})"
        else:
            return None, "网格搜索无有效模型"
    else:
        # 回退到固定参数
        order = (1, 1, 1)
        try:
            model = ARIMA(vix_series, order=order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=1)
            return forecast.iloc[0], f"固定参数: ARIMA{order}"
        except Exception as e:
            return None, f"预测错误: {str(e)}"

# 函数：TSLA 多步 ARIMA 预测（类似 VIX，支持多步）
def predict_tsla_arima(tsla_df, steps=5, enable_grid=True, p_max=2, d_max=1, q_max=2):
    if len(tsla_df) < 20:  # 需要足够数据点
        return None, None, "数据不足，无法预测"
    
    tsla_series = tsla_df['TSLA'].dropna()
    
    if enable_grid:
        # 动态网格搜索最佳ARIMA参数
        p_range = range(0, p_max + 1)
        d_range = range(0, d_max + 1)
        q_range = range(0, q_max + 1)
        param_combinations = list(product(p_range, d_range, q_range))
        
        best_aic = float("inf")
        best_order = None
        best_forecast = None
        
        for order in param_combinations:
            try:
                model = ARIMA(tsla_series, order=order)
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = order
                    forecast = fitted_model.forecast(steps=steps)
                    best_forecast = forecast
            except:
                continue
        
        if best_forecast is not None:
            return best_forecast, f"最佳参数: ARIMA{best_order} (AIC: {best_aic:.2f})", None
        else:
            return None, None, "网格搜索无有效模型"
    else:
        # 回退到固定参数
        order = (1, 1, 1)
        try:
            model = ARIMA(tsla_series, order=order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=steps)
            return forecast, f"固定参数: ARIMA{order}", None
        except Exception as e:
            return None, None, f"预测错误: {str(e)}"

# 函数：基于VIX-TSLA前N分钟升跌幅相关性预测下一分钟TSLA
def predict_next_tsla(vix_df, tsla_df, next_vix, corr_window):
    if len(vix_df) < corr_window or len(tsla_df) < corr_window or next_vix is None:
        return None, "数据不足，无法预测"
    
    # 取最近corr_window个数据点
    recent_vix = vix_df.tail(corr_window)['VIX']
    recent_tsla = tsla_df.tail(corr_window)['TSLA']
    
    # 计算分钟级百分比变化
    vix_pct = recent_vix.pct_change().dropna()
    tsla_pct = recent_tsla.pct_change().dropna()
    
    if len(vix_pct) < 2 or len(tsla_pct) < 2:
        return None, "变化数据不足，无法计算相关性"
    
    # 计算相关系数
    correlation = vix_pct.corr(tsla_pct)
    
    # 使用简单线性回归：TSLA_pct ~ VIX_pct
    X = sm.add_constant(vix_pct.values)
    model = sm.OLS(tsla_pct.values, X).fit()
    beta = model.params[1]  # 斜率
    
    # 预测VIX下一分钟变化（百分比）
    current_vix = vix_df['VIX'].iloc[-1]
    vix_delta_pct = ((next_vix - current_vix) / current_vix) * 100 if current_vix != 0 else 0
    
    # 预测TSLA变化（百分比）
    tsla_delta_pct = beta * vix_delta_pct
    
    # 预测TSLA价格
    current_tsla = tsla_df['TSLA'].iloc[-1]
    next_tsla = current_tsla * (1 + tsla_delta_pct / 100)
    
    pred_msg = f"相关性: {correlation:.3f}, Beta: {beta:.3f}, VIX变化预测: {vix_delta_pct:+.2f}%"
    
    return next_tsla, pred_msg

# 主循环：实时更新
placeholder = st.empty()

while True:
    data = fetch_data(sp500_trend_days, ma_window, corr_window)
    
    with placeholder.container():
        # 显示当前时间
        st.metric("更新时间", data['timestamp'])
        
        # 显示指标（包含实时升跌幅）
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VIX 指数", f"{data['vix']:.2f}" if data['vix'] else "N/A", 
                      f"{data['vix_change_pct']:+.2f}%" if data['vix'] else "N/A")
        with col2:
            st.metric("SP500 指数", f"{data['sp500']:.2f}" if data['sp500'] else "N/A", 
                      f"{data['sp500_change_pct']:+.2f}%" if data['sp500'] else "N/A")
        with col3:
            st.metric("TSLA 股价", f"{data['tsla']:.2f}" if data['tsla'] else "N/A", 
                      f"{data['tsla_change_pct']:+.2f}%" if data['tsla'] else "N/A")
        
        # SP500 趋势
        st.metric("SP500 趋势 (%)", f"{data['sp500_trend']:.2f}%")
        
        # VIX 实时走势图（添加MA趋势线）
        if not data['vix_df'].empty:
            st.subheader(f"VIX 实时走势图 (最近1天分钟数据，MA{ma_window}趋势线)")
            st.line_chart(data['vix_df'])
            st.caption(f"VIX 当日变化: {data['vix_change_pct']:+.2f}% (相对于开盘)")
        
        # SP500 实时走势图（添加MA趋势线）
        if not data['sp500_df'].empty:
            st.subheader(f"SP500 实时走势图 (最近1天分钟数据，MA{ma_window}趋势线)")
            st.line_chart(data['sp500_df'])
            st.caption(f"SP500 当日变化: {data['sp500_change_pct']:+.2f}% (相对于开盘)")
        
        # TSLA 实时走势图（添加MA趋势线）
        if not data['tsla_df'].empty:
            st.subheader(f"TSLA 实时走势图 (最近1天分钟数据，MA{ma_window}趋势线)")
            st.line_chart(data['tsla_df'])
            st.caption(f"TSLA 当日变化: {data['tsla_change_pct']:+.2f}% (相对于开盘)")
        
        # VIX 下一分钟预测（优化版）
        next_vix, pred_msg = predict_next_vix(data['vix_df'], enable_grid_search_vix, p_max_vix, d_max_vix, q_max_vix)
        if next_vix is not None:
            delta = next_vix - data['vix']
            trend = "上涨" if delta > 0 else "下跌" if delta < 0 else "持平"
            st.metric(f"下一分钟 VIX 预测 ({trend})", f"{next_vix:.2f}", f"{delta:+.2f}")
            st.info(pred_msg)
        else:
            st.warning(pred_msg)
        
        # TSLA 多步 ARIMA 预测（类似 VIX，支持多步）
        tsla_forecast, tsla_arima_msg, tsla_error = predict_tsla_arima(data['tsla_df'], tsla_forecast_steps, enable_grid_search_tsla, p_max_tsla, d_max_tsla, q_max_tsla)
        if tsla_forecast is not None and tsla_error is None:
            # 显示整体趋势
            overall_delta = tsla_forecast.iloc[-1] - data['tsla']
            overall_trend = "上涨" if overall_delta > 0 else "下跌" if overall_delta < 0 else "持平"
            st.metric(f"{tsla_forecast_steps}分钟 TSLA ARIMA 预测 ({overall_trend})", f"{tsla_forecast.iloc[-1]:.2f}", f"{overall_delta:+.2f}")
            st.info(tsla_arima_msg)
            
            # 显示预测图表
            st.subheader(f"TSLA ARIMA 多步预测 (未来 {tsla_forecast_steps} 分钟)")
            forecast_df = pd.DataFrame({
                '分钟': range(1, tsla_forecast_steps + 1),
                '预测价格': tsla_forecast.values
            })
            st.line_chart(forecast_df.set_index('分钟'))
        else:
            st.warning(tsla_error or tsla_arima_msg)
        
        # TSLA 下一分钟相关性预测（基于VIX-TSLA）
        if enable_tsla_corr_predict:
            next_tsla_corr, tsla_corr_msg = predict_next_tsla(data['vix_df'], data['tsla_df'], next_vix, corr_window)
            if next_tsla_corr is not None:
                delta_tsla_corr = next_tsla_corr - data['tsla']
                trend_tsla_corr = "上涨" if delta_tsla_corr > 0 else "下跌" if delta_tsla_corr < 0 else "持平"
                st.metric(f"下一分钟 TSLA 相关性预测 ({trend_tsla_corr})", f"{next_tsla_corr:.2f}", f"{delta_tsla_corr:+.2f}")
                st.info(tsla_corr_msg)
            else:
                st.warning(tsla_corr_msg)
        
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
4. **VIX 预测优化**：启用动态网格搜索以自动选择最佳 ARIMA 参数，提高预测准确度（基于当前数据的最低 AIC）。可调整参数范围以平衡速度与准确度。
5. **TSLA ARIMA 多步预测**：新增多步 ARIMA 模型预测未来 N 分钟 TSLA 股价（默认5步），显示整体趋势（上涨/下跌/持平）和预测图表。类似于 VIX 预测，但扩展到多步。
6. **TSLA 相关性预测**：基于前N分钟VIX和TSLA升跌幅的相关性，使用线性回归（OLS）预测下一分钟TSLA价格。相关性计算使用Pearson系数，Beta为回归斜率。
7. **图表改进**：添加可调节窗口期的移动平均线 (MA) 趋势线，帮助突出当前趋势方向。调整窗口期以平滑不同程度的趋势。
8. **实时数据与升跌幅**：在指标和图表下方显示相对于当天开盘的实时升跌百分比，便于快速识别趋势。
9. **注意**：这仅为教育性示例，非投资建议。实际交易需谨慎，考虑风险。预测模型基于历史短期数据，市场波动性高，准确度有限。多步预测累积误差可能更大。
""")

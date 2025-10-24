import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ==================== 页面配置 ====================
st.set_page_config(page_title="VIX & SP500 监控 + TSLA 买卖建议", layout="wide")
st.title("VIX & SP500 实时监控与 TSLA 买卖建议")

# ==================== 侧边栏参数 ====================
st.sidebar.header("设置")
refresh_interval = st.sidebar.slider("刷新间隔（秒）", 10, 300, 60, 5)

st.sidebar.header("买卖策略参数")
vix_threshold_high = st.sidebar.slider("VIX 高阈值（恐慌卖出）", 15.0, 50.0, 30.0)
vix_threshold_low = st.sidebar.slider("VIX 低阈值（安全买入）", 10.0, 25.0, 15.0)
sp500_trend_days = st.sidebar.slider("SP500 趋势天数", 5, 20, 10)

st.sidebar.header("VIX 预测参数")
enable_grid_search_vix = st.sidebar.checkbox("启用动态参数优化 (VIX ARIMA网格搜索)", value=True)
p_max_vix = st.sidebar.slider("VIX ARIMA p 最大值", 0, 3, 2)
d_max_vix = st.sidebar.slider("VIX ARIMA d 最大值", 0, 2, 1)
q_max_vix = st.sidebar.slider("VIX ARIMA q 最大值", 0, 3, 2)

st.sidebar.header("TSLA ARIMA 预测参数")
enable_grid_search_tsla = st.sidebar.checkbox("启用动态参数优化 (TSLA ARIMA网格搜索)", value=True)
p_max_tsla = st.sidebar.slider("TSLA ARIMA p 最大值", 0, 3, 2)
d_max_tsla = st.sidebar.slider("TSLA ARIMA d 最大值", 0, 2, 1)
q_max_tsla = st.sidebar.slider("TSLA ARIMA q 最大值", 0, 3, 2)
tsla_forecast_steps = st.sidebar.slider("TSLA 多步预测步数 (分钟)", 1, 10, 5)

st.sidebar.header("TSLA 相关性预测参数")
enable_tsla_corr_predict = st.sidebar.checkbox("启用基于VIX-TSLA相关性的TSLA预测", value=True)
corr_window = st.sidebar.slider("相关性计算窗口 (分钟)", 5, 30, 5)

st.sidebar.header("图表优化")
ma_window = st.sidebar.slider("移动平均窗口期 (用于趋势线)", 3, 20, 5)

st.sidebar.header("调试")
debug_mode = st.sidebar.checkbox("启用调试模式 (显示数据长度等信息)", value=False)

# 策略说明
st.sidebar.markdown("""
### 策略逻辑
- **买入建议**：VIX < 低阈值 且 SP500 过去 N 天上涨 > 2%。
- **卖出建议**：VIX > 高阈值 或 SP500 过去 N 天下跌 > 2%。
- **持有**：其他情况。
""")

# ==================== 辅助函数 ====================
def safe_last(series_or_df, col=None):
    """安全获取最后一条数据"""
    if isinstance(series_or_df, pd.Series):
        return series_or_df.iloc[-1] if len(series_or_df) > 0 else None
    elif isinstance(series_or_df, pd.DataFrame):
        if col is None or col not in series_or_df.columns:
            return None
        return series_or_df[col].iloc[-1] if len(series_or_df) > 0 else None
    return None

def get_d_order(series, max_d=2):
    """使用 ADF 检验自动确定差分阶数 d"""
    d = 0
    temp = series.copy()
    for _ in range(max_d + 1):
        if len(temp.dropna()) < 10:
            break
        result = adfuller(temp.dropna(), max_lags=1)
        if result[1] < 0.05:  # p-value < 0.05，平稳
            return d
        temp = temp.diff().dropna()
        d += 1
    return min(d, max_d)

def fetch_data(sp500_trend_days, ma_window, corr_window):
    """获取实时数据（不缓存）"""
    try:
        # VIX 分钟数据
        vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
        current_vix = safe_last(vix, 'Close')
        vix_open = safe_last(vix, 'Open') if not vix.empty else None
        vix_change_pct = ((current_vix - vix_open) / vix_open) * 100 if vix_open and vix_open != 0 else 0
        vix_df = vix[['Close']].copy() if not vix.empty else pd.DataFrame()
        if not vix_df.empty:
            vix_df.columns = ['VIX']
            vix_df['VIX_MA'] = vix_df['VIX'].rolling(window=ma_window, min_periods=1).mean()
        else:
            vix_df = pd.DataFrame(columns=['VIX', 'VIX_MA'])

        # SP500 分钟数据
        sp500_min = yf.Ticker("^GSPC").history(period="1d", interval="1m")
        current_sp500_min = safe_last(sp500_min, 'Close')
        sp500_open = safe_last(sp500_min, 'Open') if not sp500_min.empty else None
        sp500_change_pct = ((current_sp500_min - sp500_open) / sp500_open) * 100 if sp500_open and sp500_open != 0 else 0
        sp500_df = sp500_min[['Close']].copy() if not sp500_min.empty else pd.DataFrame()
        if not sp500_df.empty:
            sp500_df.columns = ['SP500']
            sp500_df['SP500_MA'] = sp500_df['SP500'].rolling(window=ma_window, min_periods=1).mean()
        else:
            sp500_df = pd.DataFrame(columns=['SP500', 'SP500_MA'])

        # SP500 日趋势
        sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days + 1}d", interval="1d")
        current_sp500 = safe_last(sp500, 'Close')
        sp500_trend = ((current_sp500 - sp500['Close'].iloc[0]) / sp500['Close'].iloc[0]) * 100 if not sp500.empty else 0.0

        # TSLA 分钟数据
        tsla = yf.Ticker("TSLA").history(period="1d", interval="1m")
        current_tsla = safe_last(tsla, 'Close')
        tsla_open = safe_last(tsla, 'Open') if not tsla.empty else None
        tsla_change_pct = ((current_tsla - tsla_open) / tsla_open) * 100 if tsla_open and tsla_open != 0 else 0
        tsla_df = tsla[['Close']].copy() if not tsla.empty else pd.DataFrame()
        if not tsla_df.empty:
            tsla_df.columns = ['TSLA']
            tsla_df['TSLA_MA'] = tsla_df['TSLA'].rolling(window=ma_window, min_periods=1).mean()
        else:
            tsla_df = pd.DataFrame(columns=['TSLA', 'TSLA_MA'])

    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

    return {
        'vix': current_vix, 'vix_df': vfe_df, 'vix_change_pct': vix_change_pct,
        'sp500': current_sp500, 'sp500_df': sp500_df, 'sp500_change_pct': sp500_change_pct,
        'sp500_trend': sp500_trend,
        'tsla': current_tsla, 'tsla_df': tsla_df, 'tsla_change_pct': tsla_change_pct,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def predict_next_vix(vix_df, enable_grid=True, p_max=2, d_max=1, q_max=2):
    if len(vix_df) < 5:
        return None, "数据不足（<5）"
    vix_series = vix_df['VIX'].dropna()
    if enable_grid:
        d_opt = get_d_order(vix_series, d_max)
        p_range = range(0, p_max + 1)
        d_range = [d_opt]
        q_range = range(0, q_max + 1)
        best_aic = float("inf")
        best_order = None
        best_forecast = None
        for p, d, q in product(p_range, d_range, q_range):
            try:
                model = ARIMA(vix_series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_forecast = fitted.forecast(steps=1).iloc[0]
            except:
                continue
        if best_forecast is not None:
            return best_forecast, f"最佳参数: ARIMA{best_order} (AIC: {best_aic:.1f})"
    else:
        try:
            model = ARIMA(vix_series, order=(1,1,1))
            fitted = model.fit()
            return fitted.forecast(steps=1).iloc[0], "固定参数: ARIMA(1,1,1)"
        except Exception as e:
            return None, f"预测失败: {e}"
    return None, "无有效模型"

def predict_tsla_arima(tsla_df, steps=5, enable_grid=True, p_max=2, d_max=1, q_max=2):
    if len(tsla_df) < 5:
        return None, None, "数据不足"
    tsla_series = tsla_df['TSLA'].dropna()
    if enable_grid:
        d_opt = get_d_order(tsla_series, d_max)
        p_range = range(0, p_max + 1)
        d_range = [d_opt]
        q_range = range(0, q_max + 1)
        best_aic = float("inf")
        best_order = None
        best_forecast = None
        for p, d, q in product(p_range, d_range, q_range):
            try:
                model = ARIMA(tsla_series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_forecast = fitted.forecast(steps=steps)
            except:
                continue
        if best_forecast is not None:
            return best_forecast, f"最佳参数: ARIMA{best_order} (AIC: {best_aic:.1f})", None
    else:
        try:
            model = ARIMA(tsla_series, order=(1,1,1))
            fitted = model.fit()
            return fitted.forecast(steps=steps), "固定参数: ARIMA(1,1,1)", None
        except Exception as e:
            return None, None, f"预测失败: {e}"
    return None, None, "无有效模型"

def predict_next_tsla(vix_df, tsla_df, next_vix, corr_window):
    if len(vix_df) < corr_window or len(tsla_df) < corr_window or next_vix is None:
        return None, "数据不足"
    recent_vix = vix_df['VIX'].tail(corr_window)
    recent_tsla = tsla_df['TSLA'].tail(corr_window)
    vix_pct = recent_vix.pct_change().dropna()
    tsla_pct = recent_tsla.pct_change().dropna()
    df_corr = pd.concat([vix_pct, tsla_pct], axis=1).dropna()
    if len(df_corr) < 2:
        return None, "变化数据不足"
    vix_p, tsla_p = df_corr.iloc[:, 0], df_corr.iloc[:, 1]
    X = sm.add_constant(vix_p)
    model = sm.OLS(tsla_p, X).fit()
    beta = model.params[1]
    current_vix = vix_df['VIX'].iloc[-1]
    vix_delta_pct = ((next_vix - current_vix) / current_vix) * 100 if current_vix != 0 else 0
    tsla_delta_pct = beta * vix_delta_pct
    current_tsla = tsla_df['TSLA'].iloc[-1]
    next_tsla = current_tsla * (1 + tsla_delta_pct / 100)
    correlation = vix_p.corr(tsla_p)
    return next_tsla, f"相关性: {correlation:.3f}, Beta: {beta:.3f}, VIX变化: {vix_delta_pct:+.2f}%"

# ==================== 渲染函数 ====================
def render_dashboard(data):
    if data is None:
        st.error("数据加载失败，请检查网络或市场时间。")
        return

    placeholder = st.empty()
    with placeholder.container():
        st.metric("更新时间", data['timestamp'])

        if debug_mode:
            st.caption(f"TSLA: {len(data['tsla_df'])} | VIX: {len(data['vix_df'])} | SP500: {len(data['sp500_df'])}")

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

        st.metric("SP500 趋势 (%)", f"{data['sp500_trend']:.2f}%")

        # 图表
        if not data['vix_df'].empty:
            st.subheader(f"VIX 实时走势图 (MA{ma_window})")
            st.line_chart(data['vix_df'])
            st.caption(f"当日变化: {data['vix_change_pct']:+.2f}%")

        if not data['sp500_df'].empty:
            st.subheader(f"SP500 实时走势图 (MA{ma_window})")
            st.line_chart(data['sp500_df'])

        if not data['tsla_df'].empty:
            st.subheader(f"TSLA 实时走势图 (MA{ma_window})")
            st.line_chart(data['tsla_df'])
        else:
            st.warning("TSLA 数据为空（非交易时间？）")

        # VIX 预测
        next_vix, vix_msg = predict_next_vix(data['vix_df'], enable_grid_search_vix, p_max_vix, d_max_vix, q_max_vix)
        if next_vix is not None and data['vix']:
            delta = next_vix - data['vix']
            trend = "上涨" if delta > 0 else "下跌" if delta < 0 else "持平"
            st.metric(f"下一分钟 VIX 预测 ({trend})", f"{next_vix:.2f}", f"{delta:+.2f}")
            st.info(vix_msg)
        else:
            st.warning(vix_msg or "VIX 预测失败")

        # TSLA ARIMA 多步预测
        if not data['tsla_df'].empty:
            tsla_forecast, arima_msg, err = predict_tsla_arima(data['tsla_df'], tsla_forecast_steps, enable_grid_search_tsla, p_max_tsla, d_max_tsla, q_max_tsla)
            if tsla_forecast is not None:
                overall_delta = tsla_forecast.iloc[-1] - data['tsla']
                trend = "上涨" if overall_delta > 0 else "下跌" if overall_delta < 0 else "持平"
                st.metric(f"{tsla_forecast_steps}分钟 TSLA 预测 ({trend})", f"{tsla_forecast.iloc[-1]:.2f}", f"{overall_delta:+.2f}")
                st.info(arima_msg)
                forecast_df = pd.DataFrame({'分钟': range(1, len(tsla_forecast)+1), '预测价格': tsla_forecast.values})
                st.line_chart(forecast_df.set_index('分钟'))
            else:
                st.warning(err or arima_msg)
        else:
            st.warning("TSLA 数据为空，无法预测")

        # 相关性预测
        if enable_tsla_corr_predict and next_vix is not None:
            next_tsla_corr, corr_msg = predict_next_tsla(data['vix_df'], data['tsla_df'], next_vix, corr_window)
            if next_tsla_corr is not None and data['tsla']:
                delta = next_tsla_corr - data['tsla']
                trend = "上涨" if delta > 0 else "下跌" if delta < 0 else "持平"
                st.metric(f"下一分钟 TSLA 相关性预测 ({trend})", f"{next_tsla_corr:.2f}", f"{delta:+.2f}")
                st.info(corr_msg)
            else:
                st.warning(corr_msg or "相关性预测失败")
        else:
            if not enable_tsla_corr_predict:
                st.info("相关性预测已禁用")

        # 买卖建议
        suggestion = "持有"
        if data['vix'] and data['vix'] > vix_threshold_high:
            suggestion = "卖出 TSLA"
        elif data['vix'] and data['vix'] < vix_threshold_low and data['sp500_trend'] > 2:
            suggestion = "买入 TSLA"
        elif data['sp500_trend'] < -2:
            suggestion = "卖出 TSLA"

        if "卖出" in suggestion:
            st.error(suggestion)
        elif "买入" in suggestion:
            st.success(suggestion)
        else:
            st.info(suggestion)

        # SP500 趋势表
        try:
            recent_sp500 = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days}d")
            st.subheader("SP500 最近趋势")
            st.dataframe(recent_sp500[['Close']].tail(5), use_container_width=True)
        except:
            st.caption("SP500 历史数据加载失败")

# ==================== 非交易时间提示 ====================
now = datetime.now()
is_trading_time = now.weekday() < 5 and 9*60 + 30 <= now.hour*60 + now.minute <= 16*60
if not is_trading_time:
    st.warning("当前为非美股交易时间（9:30-16:00 ET），数据可能为空或延迟。")

# ==================== 主流程 ====================
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

current_time = time.time()
if current_time - st.session_state.last_update >= refresh_interval:
    with st.spinner("正在刷新数据..."):
        data = fetch_data(sp500_trend_days, ma_window, corr_window)
    st.session_state.last_update = current_time
    render_dashboard(data)
else:
    time.sleep(1)
    st.rerun()

# 自动刷新
time.sleep(max(0, refresh_interval - (time.time() - st.session_state.last_update)))
st.rerun()

# ==================== 运行说明 ====================
st.markdown("---")
st.markdown("""
### 运行说明
1. `pip install streamlit yfinance pandas numpy statsmodels`
2. `streamlit run app.py`
3. **实时刷新**：每 {refresh_interval} 秒自动更新
4. **预测模型**：ARIMA + 相关性回归双模型
5. **非交易时间**：数据为空属正常
6. **仅供教育，非投资建议**
""".format(refresh_interval=refresh_interval))

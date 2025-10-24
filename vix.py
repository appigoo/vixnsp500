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
st.set_page_config(page_title="VIX & TSLA 实时预测仪表盘", layout="wide")
st.title("VIX & SP500 实时监控 + TSLA 买卖建议（含下一分钟预测）")

# ==================== 侧边栏参数 ====================
st.sidebar.header("刷新设置")
refresh_interval = st.sidebar.slider("刷新间隔（秒）", 10, 300, 60, 5)

st.sidebar.header("买卖策略参数")
vix_threshold_high = st.sidebar.slider("VIX 高阈值（恐慌卖出）", 15.0, 50.0, 30.0)
vix_threshold_low = st.sidebar.slider("VIX 低阈值（安全买入）", 10.0, 25.0, 15.0)
sp500_trend_days = st.sidebar.slider("SP500 趋势天数", 5, 20, 10)

st.sidebar.header("预测参数")
enable_grid_search = st.sidebar.checkbox("启用 ARIMA 网格搜索优化", value=True)
p_max = st.sidebar.slider("ARIMA p 最大值", 0, 3, 2)
d_max = st.sidebar.slider("ARIMA d 最大值", 0, 2, 1)
q_max = st.sidebar.slider("ARIMA q 最大值", 0, 3, 2)

st.sidebar.header("TSLA 下一分钟预测")
predict_next_tsla_enabled = st.sidebar.checkbox("启用 TSLA 下一分钟预测", value=True)
use_fusion = st.sidebar.checkbox("融合预测（ARIMA + 相关性回归）", value=True)
fusion_weight_arima = st.sidebar.slider("ARIMA 权重", 0.0, 1.0, 0.7, 0.1)

st.sidebar.header("相关性预测")
enable_corr_predict = st.sidebar.checkbox("启用 VIX-TSLA 相关性预测", value=True)
corr_window = st.sidebar.slider("相关性窗口 (分钟)", 5, 30, 10)

st.sidebar.header("图表优化")
ma_window = st.sidebar.slider("移动平均窗口", 3, 20, 5)

st.sidebar.header("调试")
debug_mode = st.sidebar.checkbox("调试模式", value=False)

# ==================== 辅助函数 ====================
def safe_last(df, col):
    return df[col].iloc[-1] if not df.empty and col in df.columns and len(df) > 0 else None

def get_d_order(series, max_d=2):
    """自动确定差分阶数 d，兼容 statsmodels ≥ 0.14"""
    d = 0
    temp = series.copy()
    for _ in range(max_d + 1):
        if len(temp.dropna()) < 10:
            break
        try:
            result = adfuller(temp.dropna(), autolag='AIC')  # 自动选择滞后
            if result[1] < 0.05:  # p < 0.05，平稳
                return d
        except Exception as e:
            if debug_mode:
                st.caption(f"ADF 检验失败: {e}")
            break
        temp = temp.diff().dropna()
        d += 1
    return min(d, max_d)

def arima_grid_search(series, p_max, d_max, q_max, steps=1):
    """统一网格搜索，返回最佳预测、order、msg"""
    if len(series) < 10:
        return None, None, "数据不足（<10）"
    
    d_opt = get_d_order(series, d_max)
    best_aic = float('inf')
    best_forecast = None
    best_order = None
    best_msg = "无有效模型"

    p_range = range(0, p_max + 1)
    d_range = [d_opt]
    q_range = range(0, q_max + 1)

    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(series, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_forecast = fitted.forecast(steps=steps)
        except:
            continue

    if best_forecast is not None:
        best_msg = f"ARIMA{best_order} (AIC: {best_aic:.1f})"
        return best_forecast, best_order, best_msg

    return None, None, best_msg

# ==================== 数据获取 ====================
def fetch_data():
    try:
        vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
        sp500_min = yf.Ticker("^GSPC").history(period="1d", interval="1m")
        sp500_day = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days + 1}d", interval="1d")
        tsla = yf.Ticker("TSLA").history(period="1d", interval="1m")

        # VIX
        vix_df = vix[['Close']].copy() if not vix.empty else pd.DataFrame()
        if not vix_df.empty:
            vix_df.columns = ['VIX']
            vix_df['VIX_MA'] = vix_df['VIX'].rolling(ma_window, min_periods=1).mean()
        current_vix = safe_last(vix_df, 'VIX')
        vix_open = vix['Open'].iloc[0] if not vix.empty else None
        vix_change = ((current_vix - vix_open) / vix_open) * 100 if vix_open and vix_open != 0 else 0

        # SP500
        sp500_df = sp500_min[['Close']].copy() if not sp500_min.empty else pd.DataFrame()
        if not sp500_df.empty:
            sp500_df.columns = ['SP500']
            sp500_df['SP500_MA'] = sp500_df['SP500'].rolling(ma_window, min_periods=1).mean()
        current_sp500 = safe_last(sp500_df, 'SP500')
        sp500_open = sp500_min['Open'].iloc[0] if not sp500_min.empty else None
        sp500_change = ((current_sp500 - sp500_open) / sp500_open) * 100 if sp500_open else 0
        sp500_trend = ((sp500_day['Close'].iloc[-1] - sp500_day['Close'].iloc[0]) / sp500_day['Close'].iloc[0]) * 100 if len(sp500_day) > 1 else 0

        # TSLA
        tsla_df = tsla[['Close']].copy() if not tsla.empty else pd.DataFrame()
        if not tsla_df.empty:
            tsla_df.columns = ['TSLA']
            tsla_df['TSLA_MA'] = tsla_df['TSLA'].rolling(ma_window, min_periods=1).mean()
        current_tsla = safe_last(tsla_df, 'TSLA')
        tsla_open = tsla['Open'].iloc[0] if not tsla.empty else None
        tsla_change = ((current_tsla - tsla_open) / tsla_open) * 100 if tsla_open else 0

    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

    return {
        'vix': current_vix, 'vix_df': vix_df, 'vix_change': vix_change,
        'sp500': current_sp500, 'sp500_df': sp500_df, 'sp500_change': sp500_change, 'sp500_trend': sp500_trend,
        'tsla': current_tsla, 'tsla_df': tsla_df, 'tsla_change': tsla_change,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ==================== 预测函数 ====================
def predict_next_vix(vix_df):
    series = vix_df['VIX'].dropna()
    forecast, _, msg = arima_grid_search(series, p_max, d_max, q_max, steps=1)
    return forecast.iloc[0] if forecast is not None else None, msg

def predict_next_tsla_arima(tsla_df):
    series = tsla_df['TSLA'].dropna()
    forecast, _, msg = arima_grid_search(series, p_max, d_max, q_max, steps=1)
    return forecast.iloc[0] if forecast is not None else None, msg

def predict_next_tsla_corr(vix_df, tsla_df, next_vix):
    if len(vix_df) < corr_window or len(tsla_df) < corr_window or next_vix is None:
        return None, "数据不足"
    vix_p = vix_df['VIX'].tail(corr_window).pct_change().dropna()
    tsla_p = tsla_df['TSLA'].tail(corr_window).pct_change().dropna()
    df = pd.concat([vix_p, tsla_p], axis=1).dropna()
    if len(df) < 2: return None, "变化不足"
    X = sm.add_constant(df.iloc[:, 0])
    model = sm.OLS(df.iloc[:, 1], X).fit()
    beta = model.params[1]
    curr_vix = vix_df['VIX'].iloc[-1]
    vix_delta = (next_vix - curr_vix) / curr_vix
    tsla_delta = beta * vix_delta
    curr_tsla = tsla_df['TSLA'].iloc[-1]
    return curr_tsla * (1 + tsla_delta), f"相关性: {df.corr().iloc[0,1]:.3f}, Beta: {beta:.3f}"

# ==================== 渲染仪表盘 ====================
def render_dashboard(data):
    if not data:
        st.error("数据加载失败")
        return

    st.metric("更新时间", data['timestamp'])
    if debug_mode:
        st.caption(f"VIX: {len(data['vix_df'])}, TSLA: {len(data['tsla_df'])}, SP500: {len(data['sp500_df'])}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VIX", f"{data['vix']:.2f}" if data['vix'] else "N/A", f"{data['vix_change']:+.2f}%" if data['vix'] else "")
    with col2:
        st.metric("SP500", f"{data['sp500']:.2f}" if data['sp500'] else "N/A", f"{data['sp500_change']:+.2f}%" if data['sp500'] else "")
    with col3:
        st.metric("TSLA", f"{data['tsla']:.2f}" if data['tsla'] else "N/A", f"{data['tsla_change']:+.2f}%" if data['tsla'] else "")

    st.metric("SP500 趋势", f"{data['sp500_trend']:.2f}%")

    # 图表
    for name, df in [("VIX", data['vix_df']), ("SP500", data['sp500_df']), ("TSLA", data['tsla_df'])]:
        if not df.empty:
            st.subheader(f"{name} 实时走势 (MA{ma_window})")
            st.line_chart(df)

    # === 下一分钟预测 ===
    st.subheader("下一分钟预测")
    col_vix, col_tsla = st.columns(2)

    with col_vix:
        next_vix, vix_msg = predict_next_vix(data['vix_df'])
        if next_vix and data['vix']:
            delta = next_vix - data['vix']
            color = "green" if delta > 0 else "red" if delta < 0 else "gray"
            st.markdown(f"**VIX 预测** <span style='color:{color}'>{'Up' if delta>0 else 'Down' if delta<0 else 'Flat'}</span>", unsafe_allow_html=True)
            st.metric("预测值", f"{next_vix:.2f}", f"{delta:+.2f}")
            st.caption(vix_msg)
        else:
            st.warning("VIX 预测失败")

    with col_tsla:
        if predict_next_tsla_enabled and data['tsla']:
            pred_arima, arima_msg = predict_next_tsla_arima(data['tsla_df'])
            pred_corr = None
            corr_msg = ""
            if enable_corr_predict and next_vix:
                pred_corr, corr_msg = predict_next_tsla_corr(data['vix_df'], data['tsla_df'], next_vix)

            if use_fusion and pred_arima and pred_corr:
                final_pred = fusion_weight_arima * pred_arima + (1 - fusion_weight_arima) * pred_corr
                source = "融合预测"
            elif pred_arima:
                final_pred = pred_arima
                source = "ARIMA"
            elif pred_corr:
                final_pred = pred_corr
                source = "相关性"
            else:
                final_pred = None
                source = ""

            if final_pred:
                delta = final_pred - data['tsla']
                color = "green" if delta > 0 else "red" if delta < 0 else "gray"
                st.markdown(f"**TSLA 预测 ({source})** <span style='color:{color}'>{'Up' if delta>0 else 'Down' if delta<0 else 'Flat'}</span>", unsafe_allow_html=True)
                st.metric("预测值", f"{final_pred:.2f}", f"{delta:+.2f}")
                if pred_arima: st.caption(arima_msg)
                if pred_corr: st.caption(corr_msg)
            else:
                st.warning("TSLA 预测失败")
        else:
            st.info("TSLA 预测已禁用")

    # 买卖建议
    suggestion = "持有"
    if data['vix'] and data['vix'] > vix_threshold_high:
        suggestion = "卖出 TSLA"
    elif data['vix'] and data['vix'] < vix_threshold_low and data['sp500_trend'] > 2:
        suggestion = "买入 TSLA"
    elif data['sp500_trend'] < -2:
        suggestion = "卖出 TSLA"

    if "卖出" in suggestion: st.error(suggestion)
    elif "买入" in suggestion: st.success(suggestion)
    else: st.info(suggestion)

    # SP500 趋势表
    try:
        hist = yf.Ticker("^GSPC").history(period=f"{sp500_trend_days}d")[['Close']].tail(5)
        st.dataframe(hist, use_container_width=True)
    except: pass

# ==================== 主循环 ====================
now = datetime.now()
is_trading = now.weekday() < 5 and 9*60 + 30 <= now.hour*60 + now.minute <= 16*60
if not is_trading:
    st.warning("非交易时间（美东 9:30-16:00），数据可能为空")

if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

if time.time() - st.session_state.last_update >= refresh_interval:
    with st.spinner("正在刷新数据..."):
        data = fetch_data()
    st.session_state.last_update = time.time()
    render_dashboard(data)
else:
    time.sleep(1)
    st.rerun()

time.sleep(max(0, refresh_interval - (time.time() - st.session_state.last_update)))
st.rerun()

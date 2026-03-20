import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="K线形态匹配", layout="wide", page_icon="📈")
st.markdown("""
<style>
.stApp { background-color: #0d1117; color: white; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 工具：统一格式化交易对
# ─────────────────────────────────────────
def normalize_symbol(symbol: str, exchange: str) -> str:
    """
    输入任意格式如 ethusdt / ETH-USDT / eth_usdt，
    自动转成各交易所需要的格式
    """
    # 先去掉分隔符，全部大写
    raw = symbol.upper().replace("-", "").replace("_", "").replace("/", "")
    # 常见 quote 币列表，从长到短排，避免误切
    quotes = ["USDT","BUSD","USDC","BTC","ETH","BNB","USD"]
    base, quote = raw, "USDT"
    for q in quotes:
        if raw.endswith(q):
            base  = raw[: -len(q)]
            quote = q
            break
    if exchange == "binance":
        return base + quote                  # ETHUSDT
    elif exchange == "bybit":
        return base + quote                  # ETHUSDT
    elif exchange == "okx":
        return base + "-" + quote           # ETH-USDT
    return base + quote

# ─────────────────────────────────────────
# 1. 拉取K线（带错误提示）
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_klines(symbol_raw: str, interval: str, limit: int, exchange: str):
    symbol = normalize_symbol(symbol_raw, exchange)
    try:
        if exchange == "binance":
            url    = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            resp   = requests.get(url, params=params, timeout=10)
            data   = resp.json()
            if isinstance(data, dict) and "code" in data:
                raise ValueError(f"Binance 错误: {data.get('msg', data)}")
            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
            ])[["time","open","high","low","close","volume"]]

        elif exchange == "bybit":
            # Bybit interval 映射
            iv_map = {"1h":"60","2h":"120","4h":"240","6h":"360","12h":"720","1d":"D"}
            iv     = iv_map.get(interval, interval.replace("h",""))
            url    = "https://api.bybit.com/v5/market/kline"
            params = {"symbol": symbol, "interval": iv, "limit": limit, "category": "spot"}
            resp   = requests.get(url, params=params, timeout=10)
            body   = resp.json()
            if body.get("retCode", 0) != 0:
                raise ValueError(f"Bybit 错误: {body.get('retMsg', body)}")
            data = body["result"]["list"]
            if not data:
                raise ValueError(f"Bybit 未返回数据，请确认交易对 {symbol} 存在")
            df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","turnover"]
                              )[["time","open","high","low","close","volume"]]
            df = df.iloc[::-1].reset_index(drop=True)

        elif exchange == "okx":
            # OKX interval 映射
            iv_map = {"1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H","1d":"1D"}
            iv     = iv_map.get(interval, interval.upper())
            url    = "https://www.okx.com/api/v5/market/history-candles"
            params = {"instId": symbol, "bar": iv, "limit": min(limit, 300)}
            resp   = requests.get(url, params=params, timeout=10)
            body   = resp.json()
            if body.get("code") != "0":
                raise ValueError(f"OKX 错误: {body.get('msg', body)}")
            data = body["data"]
            if not data:
                raise ValueError(f"OKX 未返回数据，请确认交易对 {symbol} 存在")
            df = pd.DataFrame(data, columns=[
                "time","open","high","low","close","volume","volCcy","volCcyQuote","confirm"
            ])[["time","open","high","low","close","volume"]]
            df = df.iloc[::-1].reset_index(drop=True)

        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
        if len(df) < 100:
            raise ValueError(f"数据量不足（仅 {len(df)} 根），请换交易对或周期")
        return df.reset_index(drop=True), symbol, None

    except Exception as e:
        return None, symbol, str(e)

# ─────────────────────────────────────────
# 2. 计算特征
# ─────────────────────────────────────────
def compute_features(df):
    close = df["close"]; high = df["high"]
    low   = df["low"];   open_ = df["open"]; vol = df["volume"]
    ret       = close.pct_change().fillna(0)
    ema9      = EMAIndicator(close, window=9).ema_indicator()
    ema9_dev  = ((close - ema9) / ema9).fillna(0)
    ema21     = EMAIndicator(close, window=21).ema_indicator()
    ema21_dev = ((close - ema21) / ema21).fillna(0)
    macd_hist = MACD(close).macd_diff().fillna(0) / (close + 1e-9)
    rsi       = RSIIndicator(close, window=14).rsi().fillna(50) / 100.0
    bb_pct    = BollingerBands(close, window=20, window_dev=2).bollinger_pband().fillna(0.5)
    vol_ret   = vol.pct_change().fillna(0).clip(-3, 3)
    body      = (close - open_) / (high - low).replace(0, 1e-9)
    return pd.DataFrame({
        "ret": ret, "ema9_dev": ema9_dev, "ema21_dev": ema21_dev,
        "macd_hist": macd_hist, "rsi": rsi, "bb_pct": bb_pct,
        "vol_ret": vol_ret, "body": body,
    })

# ─────────────────────────────────────────
# 3. 匹配核心
# ─────────────────────────────────────────
def match_pattern(df, features, match_len=20, top_n=5, future_len=20):
    feat_matrix = features.values
    n = len(feat_matrix)
    template = feat_matrix[n - match_len:]
    def zscore(arr):
        std = arr.std(axis=0); std[std == 0] = 1
        return (arr - arr.mean(axis=0)) / std
    template_z = zscore(template).flatten()
    scores = []
    search_end = n - match_len * 3
    for i in range(match_len + 50, search_end):
        window   = feat_matrix[i - match_len: i]
        window_z = zscore(window).flatten()
        sim      = 1 - cosine(template_z, window_z)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected, used_idx = [], []
    for idx, sim in scores:
        if all(abs(idx - u) > match_len * 2 for u in used_idx):
            selected.append((idx, sim)); used_idx.append(idx)
        if len(selected) >= top_n:
            break
    results = []
    for idx, sim in selected:
        hist   = df.iloc[idx - match_len: idx].copy()
        future = df.iloc[idx: idx + future_len].copy()
        results.append({
            "match_end_time": df.iloc[idx]["time"],
            "similarity":     sim,
            "history":        hist,
            "future":         future,
        })
    return results

# ─────────────────────────────────────────
# 4. Plotly K线图
# ─────────────────────────────────────────
def make_candle_fig(data, title, show_divider=False, match_len=20):
    data = data.reset_index(drop=True)
    fig  = go.Figure()
    if show_divider:
        hist   = data.iloc[:match_len]
        future = data.iloc[match_len:]
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["open"], high=hist["high"],
            low=hist["low"], close=hist["close"], name="历史",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        ))
        if len(future):
            fig.add_trace(go.Candlestick(
                x=future.index, open=future["open"], high=future["high"],
                low=future["low"], close=future["close"], name="后续走势",
                increasing_line_color="#ffa726", decreasing_line_color="#ab47bc",
                increasing_fillcolor="#ffa726",  decreasing_fillcolor="#ab47bc",
            ))
        fig.add_vline(x=match_len - 0.5, line_dash="dash",
                      line_color="white", line_width=1.5)
    else:
        fig.add_trace(go.Candlestick(
            x=data.index, open=data["open"], high=data["high"],
            low=data["low"], close=data["close"], name="当前",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",  decreasing_fillcolor="#ef5350",
        ))
    fig.update_layout(
        title=title, title_font_color="white", title_font_size=13,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=False, color="#888"),
        yaxis=dict(showgrid=True, gridcolor="#1e2530", color="#888"),
        xaxis_rangeslider_visible=False,
        legend=dict(font_color="white", bgcolor="#0d1117"),
        margin=dict(l=50, r=20, t=50, b=30),
        height=320,
    )
    return fig

# ─────────────────────────────────────────
# 5. 主界面
# ─────────────────────────────────────────
st.title("📈 K线形态匹配系统")

with st.sidebar:
    st.header("⚙️ 参数设置")
    exchange   = st.selectbox("交易所", ["binance", "bybit", "okx"])
    symbol_raw = st.text_input("交易对（大小写均可）", value="BTCUSDT",
                                help="支持 btcusdt / BTC-USDT / eth_usdt 等格式")
    interval   = st.selectbox("K线周期", ["1h","2h","4h","6h","12h","1d"], index=2)
    match_len  = st.slider("模板长度（根）", 10, 50, 20)
    future_len = st.slider("后续预测长度（根）", 5, 40, 20)
    top_n      = st.slider("显示 Top N 结果", 1, 10, 5)
    run_btn    = st.button("🚀 开始匹配", use_container_width=True)

if run_btn:
    with st.spinner("📡 拉取数据中..."):
        df, symbol_used, err = fetch_klines(symbol_raw, interval, 1500, exchange)

    if err:
        st.error(f"❌ 数据获取失败：{err}")
        st.info(f"📌 实际请求的交易对：`{symbol_used}`\n\n请检查交易对名称是否在 {exchange} 上存在")
        st.stop()

    st.success(f"✅ [{exchange.upper()}] {symbol_used}  共 {len(df)} 根K线  "
               f"{df['time'].iloc[0].strftime('%Y-%m-%d')} ~ {df['time'].iloc[-1].strftime('%Y-%m-%d')}")

    with st.spinner("🔍 计算特征 & 匹配中..."):
        features = compute_features(df)
        results  = match_pattern(df, features, match_len=match_len,
                                 top_n=top_n, future_len=future_len)

    if not results:
        st.warning("⚠️ 未找到匹配结果，请尝试减小模板长度或换更长的周期")
        st.stop()

    # 当前模板
    st.subheader("📍 当前模板")
    current = df.iloc[-match_len:].copy()
    start_t = current.iloc[0]["time"].strftime("%Y-%m-%d")
    end_t   = current.iloc[-1]["time"].strftime("%Y-%m-%d")
    st.plotly_chart(make_candle_fig(current, f"当前模板  {start_t} ~ {end_t}"),
                    use_container_width=True)

    st.divider()
    st.subheader("🏆 历史匹配结果")

    summary = pd.DataFrame([{
        "排名":   f"#{i+1}",
        "相似度": f"{r['similarity']*100:.1f}%",
        "历史时间": r['match_end_time'].strftime("%Y-%m-%d %H:%M"),
    } for i, r in enumerate(results)])
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.divider()

    for i, res in enumerate(results):
        sim_pct  = res["similarity"] * 100
        time_str = res["match_end_time"].strftime("%Y-%m-%d %H:%M")
        combined = pd.concat([res["history"], res["future"]], ignore_index=True)
        title    = f"#{i+1}  相似度 {sim_pct:.1f}%  |  历史时间: {time_str}  |  橙色=后续走势"
        st.plotly_chart(make_candle_fig(combined, title,
                        show_divider=True, match_len=match_len),
                        use_container_width=True)
else:
    st.info("👈 在左侧设置参数，点击「开始匹配」按钮运行")

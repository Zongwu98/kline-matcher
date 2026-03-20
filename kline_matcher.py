import requests
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# ── 中文字体修复 ──
def set_chinese_font():
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        try:
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
        except:
            continue
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# ─────────────────────────────────────────
# 1. 拉取K线数据
# ─────────────────────────────────────────
def fetch_klines(symbol="BTCUSDT", interval="4h", limit=1500, exchange="binance"):
    if exchange == "binance":
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        df = df[["time","open","high","low","close","volume"]]

    elif exchange == "bybit":
        url = "https://api.bybit.com/v5/market/kline"
        params = {"symbol": symbol, "interval": interval.replace("h",""), "limit": limit, "category": "spot"}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()["result"]["list"]
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","turnover"])
        df = df[["time","open","high","low","close","volume"]]
        df = df.iloc[::-1].reset_index(drop=True)

    elif exchange == "okx":
        url = "https://www.okx.com/api/v5/market/history-candles"
        inst = symbol[:3] + "-" + symbol[3:]
        params = {"instId": inst, "bar": interval.upper(), "limit": limit}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()["data"]
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
        df = df[["time","open","high","low","close","volume"]]
        df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────
# 2. 计算8维特征
# ─────────────────────────────────────────
def compute_features(df):
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]
    vol   = df["volume"]

    ret        = close.pct_change().fillna(0)
    ema9       = EMAIndicator(close, window=9).ema_indicator()
    ema9_dev   = ((close - ema9) / ema9).fillna(0)
    ema21      = EMAIndicator(close, window=21).ema_indicator()
    ema21_dev  = ((close - ema21) / ema21).fillna(0)
    macd       = MACD(close)
    macd_hist  = macd.macd_diff().fillna(0) / (close + 1e-9)
    rsi        = RSIIndicator(close, window=14).rsi().fillna(50) / 100.0
    bb         = BollingerBands(close, window=20, window_dev=2)
    bb_pct     = bb.bollinger_pband().fillna(0.5)
    vol_ret    = vol.pct_change().fillna(0).clip(-3, 3)
    rng        = (high - low).replace(0, 1e-9)
    body       = (close - open_) / rng

    return pd.DataFrame({
        "ret": ret, "ema9_dev": ema9_dev, "ema21_dev": ema21_dev,
        "macd_hist": macd_hist, "rsi": rsi, "bb_pct": bb_pct,
        "vol_ret": vol_ret, "body": body,
    })


# ─────────────────────────────────────────
# 3. 匹配核心（修复去重逻辑）
# ─────────────────────────────────────────
def match_pattern(df, features, match_len=20, top_n=5, future_len=20):
    feat_matrix = features.values
    n = len(feat_matrix)
    template = feat_matrix[n - match_len:]

    def zscore(arr):
        std = arr.std(axis=0)
        std[std == 0] = 1
        return (arr - arr.mean(axis=0)) / std

    template_z = zscore(template).flatten()

    scores = []
    # 排除最近 match_len*3 根，避免匹配到近期重叠数据
    search_end = n - match_len * 3
    for i in range(match_len + 50, search_end):
        window = feat_matrix[i - match_len: i]
        window_z = zscore(window).flatten()
        sim = 1 - cosine(template_z, window_z)
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    used_idx = []
    for idx, sim in scores:
        if all(abs(idx - u) > match_len * 2 for u in used_idx):
            selected.append((idx, sim))
            used_idx.append(idx)
        if len(selected) >= top_n:
            break

    results = []
    for idx, sim in selected:
        hist   = df.iloc[idx - match_len: idx].copy()
        future = df.iloc[idx: idx + future_len].copy()
        results.append({
            "match_end_idx":  idx,
            "match_end_time": df.iloc[idx]["time"],
            "similarity":     sim,
            "history":        hist,
            "future":         future,
        })
    return results


# ─────────────────────────────────────────
# 4. 可视化（修复显示所有结果）
# ─────────────────────────────────────────
def plot_candles(ax, data, title, show_divider=False, match_len=20):
    ax.set_facecolor('#0d1117')
    ax.set_title(title, color='white', fontsize=8, pad=5)
    ax.tick_params(colors='#888888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    data = data.reset_index(drop=True)
    for i, row in data.iterrows():
        is_future = show_divider and i >= match_len
        if is_future:
            color = '#ffa726' if row['close'] >= row['open'] else '#ab47bc'
        else:
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height == 0:
            body_height = (row['high'] - row['low']) * 0.1
        ax.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.9)

    if show_divider:
        ax.axvline(x=match_len - 0.5, color='#ffffff', linewidth=1,
                   linestyle='--', alpha=0.6, label='匹配结束')
    ax.set_xlim(-1, len(data))


def plot_results(df, results, match_len=20, future_len=20):
    n_rows = len(results) + 1
    fig_height = 3.5 * n_rows
    fig = plt.figure(figsize=(16, fig_height))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.7,
                           top=0.95, bottom=0.03, left=0.06, right=0.97)

    # 当前模板
    ax0 = fig.add_subplot(gs[0])
    current = df.iloc[-match_len:].copy()
    start_t = current.iloc[0]["time"].strftime("%Y-%m-%d")
    end_t   = current.iloc[-1]["time"].strftime("%Y-%m-%d")
    plot_candles(ax0, current, f"📍 当前模板（最近 {match_len} 根K线  {start_t} ~ {end_t}）")

    # 历史匹配
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[i + 1])
        combined = pd.concat([res["history"], res["future"]], ignore_index=True)
        sim_pct  = res["similarity"] * 100
        time_str = res["match_end_time"].strftime("%Y-%m-%d %H:%M")
        title = f"#{i+1}  相似度 {sim_pct:.1f}%  |  历史时间: {time_str}  |  橙色=后续走势"
        plot_candles(ax, combined, title, show_divider=True, match_len=match_len)

    fig.suptitle("BTC K线形态匹配结果", color='white', fontsize=13, fontweight='bold')
    plt.savefig("kline_match_result.png", dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    print("\n✅ 图表已保存为 kline_match_result.png")
    plt.show()


# ─────────────────────────────────────────
# 5. 主程序
# ─────────────────────────────────────────
if __name__ == "__main__":
    EXCHANGE   = "binance"
    SYMBOL     = "BTCUSDT"
    INTERVAL   = "4h"
    MATCH_LEN  = 20
    FUTURE_LEN = 20
    TOP_N      = 5

    print(f"📡 从 {EXCHANGE} 拉取 {SYMBOL} {INTERVAL} K线...")
    df = fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=1500, exchange=EXCHANGE)
    print(f"✅ 获取 {len(df)} 根K线  {df['time'].iloc[0]} ~ {df['time'].iloc[-1]}")

    print("🔧 计算特征指标...")
    features = compute_features(df)

    print(f"🔍 开始匹配（模板={MATCH_LEN}根，返回Top{TOP_N}）...")
    results = match_pattern(df, features, match_len=MATCH_LEN, top_n=TOP_N, future_len=FUTURE_LEN)

    print("\n🏆 匹配结果：")
    for i, r in enumerate(results):
        print(f"  #{i+1} 相似度={r['similarity']*100:.1f}%  时间={r['match_end_time']}")

    print("\n📊 绘制图表...")
    plot_results(df, results, match_len=MATCH_LEN, future_len=FUTURE_LEN)
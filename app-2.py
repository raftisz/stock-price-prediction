"""
Stock Price Prediction App
Streamlit application for predicting next-day closing prices
using Random Forest Regressor trained on 2018–2026 data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #4285f4;
    }
    .metric-label { color: #aaaaaa; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 26px; font-weight: bold; }
    .metric-sub   { color: #888888; font-size: 12px; margin-top: 2px; }
    .predict-box  {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border: 2px solid #4285f4;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
    }
    .predict-price { color: #76b900; font-size: 48px; font-weight: bold; }
    .predict-label { color: #aaaaaa; font-size: 14px; margin-bottom: 8px; }
    .disclaimer {
        background: #2a1f1f;
        border-left: 4px solid #ff4d6a;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 13px;
        color: #ccaaaa;
    }
    .feature-explain {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
    }
    .rsi-ob { color: #ff4d6a; font-weight: bold; }
    .rsi-os { color: #76b900; font-weight: bold; }
    .rsi-ok { color: #ffd166; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'JPM', 'GS']
TICKER_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.',
    'JPM':  'JPMorgan Chase & Co.',
    'GS':   'Goldman Sachs Group Inc.',
}
COLORS = {
    'AAPL': '#76b900',
    'MSFT': '#ff9900',
    'AMZN': '#4285f4',
    'JPM':  '#aaaaaa',
    'GS':   '#f4b942',
}
FEATURE_COLS = [
    'Close', 'Open', 'High', 'Low', 'Volume',
    'Returns', 'MA5', 'MA20', 'MA50',
    'Volatility', 'RSI', 'Price_Range', 'Volume_Ratio'
]
FEATURE_INFO = {
    'Close':        ('ราคาปิด',          'ราคาปิดของวันนั้น (USD)'),
    'Open':         ('ราคาเปิด',         'ราคาเปิดของวันนั้น (USD)'),
    'High':         ('ราคาสูงสุด',       'ราคาสูงสุดของวัน (USD)'),
    'Low':          ('ราคาต่ำสุด',       'ราคาต่ำสุดของวัน (USD)'),
    'Volume':       ('ปริมาณซื้อขาย',   'จำนวนหุ้นที่ซื้อขายในวันนั้น'),
    'Returns':      ('ผลตอบแทนรายวัน',  '% เปลี่ยนแปลงจากวันก่อน'),
    'MA5':          ('MA 5 วัน',         'ค่าเฉลี่ยราคาปิด 5 วันล่าสุด'),
    'MA20':         ('MA 20 วัน',        'ค่าเฉลี่ยราคาปิด 20 วันล่าสุด'),
    'MA50':         ('MA 50 วัน',        'ค่าเฉลี่ยราคาปิด 50 วันล่าสุด'),
    'Volatility':   ('ความผันผวน',       'SD ของ Returns 20 วัน — ยิ่งสูงยิ่งเสี่ยง'),
    'RSI':          ('RSI 14 วัน',       'Relative Strength Index (0–100)'),
    'Price_Range':  ('ช่วงราคา',         'High - Low ของวัน (USD)'),
    'Volume_Ratio': ('Volume Ratio',     'ปริมาณซื้อขายเทียบกับ MA 20 วัน'),
}

# ── Data & model helpers ──────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('StockPriceDataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_data
def create_features(df, ticker):
    data = df[df['Ticker'] == ticker].copy()
    data = data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].copy()
    data['Returns']      = data['Close'].pct_change()
    data['Price_Range']  = data['High'] - data['Low']
    data['MA5']          = data['Close'].rolling(5).mean()
    data['MA20']         = data['Close'].rolling(20).mean()
    data['MA50']         = data['Close'].rolling(50).mean()
    data['Volatility']   = data['Returns'].rolling(20).std()
    gain = data['Returns'].clip(lower=0)
    loss = (-data['Returns']).clip(lower=0)
    rs   = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)
    data['RSI']          = 100 - (100 / (1 + rs))
    data['Volume_MA']    = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / (data['Volume_MA'] + 1e-10)
    data['Target_Price'] = data['Close'].shift(-1)
    return data.dropna().reset_index(drop=True)

@st.cache_resource
def train_model(ticker, _df):
    data   = create_features(_df, ticker)
    n      = len(data)
    idx    = int(n * 0.8)
    X      = data[FEATURE_COLS]
    y      = data['Target_Price']
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X.iloc[:idx])
    X_te_s = scaler.transform(X.iloc[idx:])
    model  = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y.iloc[:idx])
    y_pred = model.predict(X_te_s)
    rmse   = np.sqrt(mean_squared_error(y.iloc[idx:], y_pred))
    r2     = r2_score(y.iloc[idx:], y_pred)
    return model, scaler, data, y_pred, y.iloc[idx:], data['Date'].iloc[idx:].reset_index(drop=True), rmse, r2

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Predictor")
    st.markdown("---")

    ticker = st.selectbox(
        "เลือกหุ้น",
        TICKERS,
        format_func=lambda t: f"{t} — {TICKER_NAMES[t]}"
    )
    st.markdown("---")
    st.markdown("### 🔧 ป้อนข้อมูลวันที่ต้องการพยากรณ์")
    st.caption("กรอกข้อมูลราคาล่าสุดเพื่อทำนายราคาปิดวันถัดไป")

    col1, col2 = st.columns(2)
    with col1:
        inp_close  = st.number_input("Close ($)",  min_value=0.01, value=150.0,  step=0.5)
        inp_high   = st.number_input("High ($)",   min_value=0.01, value=152.0,  step=0.5)
        inp_low    = st.number_input("Low ($)",    min_value=0.01, value=148.0,  step=0.5)
        inp_open   = st.number_input("Open ($)",   min_value=0.01, value=149.0,  step=0.5)
    with col2:
        inp_volume = st.number_input("Volume (M)", min_value=0.01, value=60.0,   step=1.0)
        inp_ma5    = st.number_input("MA5 ($)",    min_value=0.01, value=149.0,  step=0.5)
        inp_ma20   = st.number_input("MA20 ($)",   min_value=0.01, value=147.0,  step=0.5)
        inp_ma50   = st.number_input("MA50 ($)",   min_value=0.01, value=145.0,  step=0.5)

    inp_rsi  = st.slider("RSI (0–100)", 0.0, 100.0, 55.0, 0.5)
    inp_vol  = st.number_input("Volatility", min_value=0.0, value=0.015, step=0.001, format="%.4f")

    # ── Input validation ──────────────────────────────────────
    errors = []
    if inp_high < inp_close:
        errors.append("⚠️ High ต้องมากกว่าหรือเท่ากับ Close")
    if inp_low  > inp_close:
        errors.append("⚠️ Low ต้องน้อยกว่าหรือเท่ากับ Close")
    if inp_high < inp_low:
        errors.append("⚠️ High ต้องมากกว่า Low")
    if inp_ma5 <= 0 or inp_ma20 <= 0 or inp_ma50 <= 0:
        errors.append("⚠️ Moving Averages ต้องมากกว่า 0")

    for e in errors:
        st.error(e)

    predict_btn = st.button("🔮 พยากรณ์ราคาวันถัดไป", use_container_width=True, disabled=len(errors) > 0)
    st.markdown("---")
    st.markdown("### 📊 ตั้งค่ากราฟ")
    show_days = st.slider("แสดงข้อมูลกี่วันล่าสุด", 30, 500, 180)

# ── Load data & train ─────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ StockPriceDataset.csv — กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

with st.spinner(f"กำลังโหลดและ train โมเดล {ticker}..."):
    model, scaler, feat_data, y_pred_test, y_test, dates_test, rmse, r2 = train_model(ticker, df)

# ── Header ────────────────────────────────────────────────────
color = COLORS[ticker]
st.markdown(f"# 📈 {ticker} — {TICKER_NAMES[ticker]}")
st.caption("พยากรณ์ราคาปิดวันถัดไปด้วย Random Forest Regressor | ข้อมูลฝึก: 2018–2026")

# ── Model metrics ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
latest = feat_data['Close'].iloc[-1]
latest_rsi = feat_data['RSI'].iloc[-1]

with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ราคาปิดล่าสุด</div>
        <div class="metric-value">${latest:.2f}</div>
        <div class="metric-sub">จากชุดข้อมูล</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RMSE (ความคลาดเคลื่อนเฉลี่ย)</div>
        <div class="metric-value">${rmse:.2f}</div>
        <div class="metric-sub">ยิ่งต่ำยิ่งดี</div>
    </div>""", unsafe_allow_html=True)
with m3:
    r2_color = "#76b900" if r2 >= 0.8 else "#ffd166" if r2 >= 0.5 else "#ff4d6a"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value" style="color:{r2_color}">{r2:.4f}</div>
        <div class="metric-sub">ยิ่งใกล้ 1 ยิ่งดี</div>
    </div>""", unsafe_allow_html=True)
with m4:
    rsi_val = latest_rsi
    rsi_label = "🔴 Overbought" if rsi_val >= 70 else "🟢 Oversold" if rsi_val <= 30 else "🟡 Normal"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RSI ปัจจุบัน</div>
        <div class="metric-value">{rsi_val:.1f}</div>
        <div class="metric-sub">{rsi_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 พยากรณ์ราคา", "📊 ประสิทธิภาพโมเดล", "📚 ข้อมูล Features"])

with tab1:
    c_pred, c_feat = st.columns([1, 1])

    with c_pred:
        st.markdown("### 🔮 ผลพยากรณ์")

        if predict_btn and not errors:
            returns_input    = (inp_close - inp_ma5) / (inp_ma5 + 1e-10)
            price_range_inp  = inp_high - inp_low
            volume_ratio_inp = (inp_volume * 1e6) / ((inp_volume * 1e6) + 1e-10)

            X_input = np.array([[
                inp_close, inp_open, inp_high, inp_low,
                inp_volume * 1_000_000,
                returns_input, inp_ma5, inp_ma20, inp_ma50,
                inp_vol, inp_rsi, price_range_inp, volume_ratio_inp
            ]])
            X_scaled   = scaler.transform(X_input)
            pred_price = model.predict(X_scaled)[0]

            tree_preds = np.array([t.predict(X_scaled)[0] for t in model.estimators_])
            pred_std   = tree_preds.std()
            pred_low   = pred_price - 1.96 * pred_std
            pred_high  = pred_price + 1.96 * pred_std

            change     = pred_price - inp_close
            change_pct = (change / inp_close) * 100
            direction  = "🟢 ขึ้น" if change >= 0 else "🔴 ลง"

            st.markdown(f"""
            <div class="predict-box">
                <div class="predict-label">ราคาปิดที่คาดการณ์วันถัดไป</div>
                <div class="predict-price">${pred_price:.2f}</div>
                <div style="color:{'#76b900' if change >= 0 else '#ff4d6a'}; font-size:18px; margin-top:8px;">
                    {direction} {change:+.2f} ({change_pct:+.2f}%)
                </div>
                <div style="color:#888; font-size:13px; margin-top:12px;">
                    95% Confidence Interval<br>
                    <strong style="color:#aaa">${pred_low:.2f} – ${pred_high:.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> ผลพยากรณ์นี้เป็นเพียงการประมาณการทางสถิติจากข้อมูลในอดีต
                ไม่ใช่คำแนะนำด้านการลงทุน ราคาหุ้นจริงอาจแตกต่างจากการพยากรณ์อย่างมีนัยสำคัญ
                เนื่องจากตลาดได้รับอิทธิพลจากปัจจัยภายนอกที่โมเดลไม่ได้รับข้อมูล
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Feature ที่ใช้พยากรณ์")
            feat_imp = pd.DataFrame({
                'Feature': FEATURE_COLS,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
            fig_fi.patch.set_facecolor('#0e1117')
            ax_fi.set_facecolor('#0e1117')
            colors_fi = ['#76b900' if imp > 0.1 else '#4285f4' if imp > 0.05 else '#555'
                         for imp in feat_imp['Importance']]
            ax_fi.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1],
                       color=colors_fi[::-1], edgecolor='none')
            ax_fi.set_xlabel('Importance Score', color='white')
            ax_fi.tick_params(colors='white')
            ax_fi.spines[:].set_color('#333')
            fig_fi.tight_layout()
            st.pyplot(fig_fi)

        else:
            st.info("👈 กรอกข้อมูลในแถบซ้ายแล้วกด **พยากรณ์ราคาวันถัดไป**")

            last_pred = y_pred_test[-1]
            last_actual = float(y_test.iloc[-1])
            st.markdown(f"""
            <div class="predict-box">
                <div class="predict-label">การพยากรณ์ล่าสุดจากชุดข้อมูล Test</div>
                <div class="predict-price">${last_pred:.2f}</div>
                <div style="color:#aaa; font-size:14px; margin-top:8px;">
                    ราคาจริง: <strong style="color:#76b900">${last_actual:.2f}</strong>
                    &nbsp;|&nbsp; Error: <strong style="color:#ff9900">${abs(last_pred - last_actual):.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c_feat:
        st.markdown("### 📈 กราฟราคาล่าสุด")
        display_data = feat_data.tail(show_days)

        fig_price, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        fig_price.patch.set_facecolor('#0e1117')
        for ax in axes:
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.spines[:].set_color('#333')
            ax.grid(alpha=0.12)

        ax = axes[0]
        ax.plot(display_data['Date'], display_data['Close'], color=color,  lw=2, label='Close')
        ax.plot(display_data['Date'], display_data['MA5'],   color='#ffd166', lw=1, ls='--', label='MA5')
        ax.plot(display_data['Date'], display_data['MA20'],  color='#ef8c8c', lw=1, ls='--', label='MA20')
        ax.plot(display_data['Date'], display_data['MA50'],  color='#88ccee', lw=1, ls='--', label='MA50')
        ax.set_ylabel('Price (USD)', color='white')
        ax.legend(fontsize=7, framealpha=0.2, labelcolor='white')
        ax.set_title(f'{ticker} — Last {show_days} Days', color='white', fontsize=10)

        ax = axes[1]
        ax.plot(display_data['Date'], display_data['RSI'], color='#c77dff', lw=1.3)
        ax.axhline(70, color='#ff4d6a', ls='--', alpha=0.6, lw=1)
        ax.axhline(30, color='#76b900', ls='--', alpha=0.6, lw=1)
        ax.fill_between(display_data['Date'], display_data['RSI'], 70,
                        where=(display_data['RSI'] >= 70), alpha=0.2, color='#ff4d6a')
        ax.fill_between(display_data['Date'], display_data['RSI'], 30,
                        where=(display_data['RSI'] <= 30), alpha=0.2, color='#76b900')
        ax.set_ylabel('RSI', color='white')
        ax.set_ylim(0, 100)
        ax.text(display_data['Date'].iloc[-1], 72, 'Overbought', color='#ff4d6a', fontsize=7, ha='right')
        ax.text(display_data['Date'].iloc[-1], 25, 'Oversold',   color='#76b900', fontsize=7, ha='right')

        ax = axes[2]
        ax.fill_between(display_data['Date'], display_data['Volatility'],
                        alpha=0.7, color='#ffd166')
        ax.set_ylabel('Volatility', color='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, color='white')

        fig_price.tight_layout()
        st.pyplot(fig_price)

with tab2:
    st.markdown("### 📊 ประสิทธิภาพโมเดลบน Test Set")
    st.caption(f"Test set = 20% สุดท้ายของข้อมูล (time-based split)")

    fig_eval, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig_eval.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.grid(alpha=0.15)

    ax = axes[0]
    ax.plot(dates_test, np.array(y_test), color='white',  lw=1.8, label='Actual')
    ax.plot(dates_test, y_pred_test,      color=color,    lw=1.5, ls='--', label='Predicted')
    ax.fill_between(dates_test, np.array(y_test), y_pred_test,
                    alpha=0.15, color='#ff4d6a', label='Error')
    ax.set_title(f'Actual vs Predicted | RMSE=${rmse:.2f}  R²={r2:.4f}',
                 color='white', fontsize=10)
    ax.set_ylabel('Price (USD)', color='white')
    ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax = axes[1]
    ax.scatter(y_test, y_pred_test, alpha=0.35, color=color, s=8)
    mn = min(float(y_test.min()), y_pred_test.min())
    mx = max(float(y_test.max()), y_pred_test.max())
    ax.plot([mn, mx], [mn, mx], color='white', lw=1.5, ls='--', label='Perfect fit')
    ax.set_xlabel('Actual Price ($)', color='white')
    ax.set_ylabel('Predicted Price ($)', color='white')
    ax.set_title(f'Actual vs Predicted Scatter', color='white', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor='white')

    fig_eval.tight_layout()
    st.pyplot(fig_eval)

    residuals = np.array(y_test) - y_pred_test
    fig_res, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    fig_res.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.grid(alpha=0.15)

    axes[0].hist(residuals, bins=40, color='#4285f4', alpha=0.8, edgecolor='none')
    axes[0].axvline(0, color='white', lw=1.5, ls='--')
    axes[0].set_title(f'Residuals Distribution  mean=${residuals.mean():.2f}  std=${residuals.std():.2f}',
                      color='white', fontsize=10)
    axes[0].set_xlabel('Residual ($)', color='white')
    axes[0].set_ylabel('Count', color='white')

    axes[1].fill_between(dates_test, np.abs(residuals), alpha=0.7, color='#ff9900')
    axes[1].set_title(f'|Error| Over Time  RMSE=${rmse:.2f}', color='white', fontsize=10)
    axes[1].set_xlabel('Date', color='white')
    axes[1].set_ylabel('|Error| ($)', color='white')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig_res.tight_layout()
    st.pyplot(fig_res)

    st.markdown("#### 💡 การแปลผล Metrics")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.info(f"**RMSE = ${rmse:.2f}** หมายความว่าโมเดลพยากรณ์ราคาผิดเฉลี่ย ${rmse:.2f} ต่อวัน "
                f"({(rmse/latest*100):.1f}% ของราคาปัจจุบัน)")
    with mc2:
        r2_interp = "ดีมาก" if r2 >= 0.8 else "ปานกลาง" if r2 >= 0.5 else "ต้องปรับปรุง"
        st.info(f"**R² = {r2:.4f}** โมเดลอธิบาย Variance ของราคาได้ {r2*100:.1f}% — ระดับ{r2_interp}")

with tab3:
    st.markdown("### 📚 อธิบาย Features ทั้งหมด")
    st.caption("Features ที่โมเดลใช้ในการพยากรณ์ราคาปิดวันถัดไป")

    for feat, (thai_name, desc) in FEATURE_INFO.items():
        imp = model.feature_importances_[FEATURE_COLS.index(feat)]
        bar_w = int(imp * 400)
        st.markdown(f"""
        <div class="feature-explain">
            <strong style="color:#ffffff">{feat}</strong>
            <span style="color:#888; margin-left:8px; font-size:12px">{thai_name}</span>
            <span style="float:right; color:#ff9900; font-size:12px">Importance: {imp*100:.1f}%</span>
            <div style="background:#333; border-radius:4px; height:4px; margin:6px 0;">
                <div style="background:#4285f4; width:{bar_w}px; max-width:100%; height:4px; border-radius:4px;"></div>
            </div>
            <div style="color:#aaa; font-size:12px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📖 คำแนะนำการอ่าน RSI")
    r1, r2c, r3 = st.columns(3)
    with r1:
        st.markdown('<div style="background:#2a1f1f;padding:12px;border-radius:8px;border-left:4px solid #ff4d6a">'
                    '<strong style="color:#ff4d6a">RSI > 70 — Overbought</strong><br>'
                    '<span style="color:#ccc;font-size:13px">ราคาขึ้นเร็วเกินไป อาจเกิด correction</span></div>',
                    unsafe_allow_html=True)
    with r2c:
        st.markdown('<div style="background:#1f2a1f;padding:12px;border-radius:8px;border-left:4px solid #76b900">'
                    '<strong style="color:#76b900">RSI < 30 — Oversold</strong><br>'
                    '<span style="color:#ccc;font-size:13px">ราคาลงเร็วเกินไป อาจเกิด rebound</span></div>',
                    unsafe_allow_html=True)
    with r3:
        st.markdown('<div style="background:#1f1f2a;padding:12px;border-radius:8px;border-left:4px solid #ffd166">'
                    '<strong style="color:#ffd166">RSI 30–70 — Normal</strong><br>'
                    '<span style="color:#ccc;font-size:13px">ตลาดอยู่ในสภาวะสมดุล</span></div>',
                    unsafe_allow_html=True)

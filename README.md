# 📈 Stock Price Prediction App

พยากรณ์ราคาปิดหุ้นวันถัดไปด้วย Random Forest Regressor  
ครอบคลุมหุ้น 5 ตัว: **AAPL · MSFT · AMZN · JPM · GS**  
ข้อมูลช่วงปี **2018–2026**

---

## 🚀 Demo

👉 [stock-price-prediction-qsozmjyvb7prqzhtzyyxa3.streamlit.app](https://stock-price-prediction-qsozmjyvb7prqzhtzyyxa3.streamlit.app)

---

## 📦 หุ้นที่รองรับ

| Ticker | ชื่อบริษัท |
|--------|-----------|
| AAPL   | Apple Inc. |
| MSFT   | Microsoft Corp. |
| AMZN   | Amazon.com Inc. |
| JPM    | JPMorgan Chase & Co. |
| GS     | Goldman Sachs Group Inc. |

---

## 🧠 โมเดลและข้อมูล

- **Algorithm:** Random Forest Regressor (200 trees)
- **ข้อมูลฝึก:** 80% แรกของแต่ละหุ้น (2018–~2024)
- **ข้อมูล Test:** 20% สุดท้าย (~2024–2026)
- **Features:** Close, Open, High, Low, Volume, Returns, MA5, MA20, MA50, Volatility, RSI, Price_Range, Volume_Ratio

---

## 📁 โครงสร้างไฟล์

```
stock-price-prediction/
├── app.py                  # Streamlit application
├── StockPriceDataset.csv   # ข้อมูลราคาหุ้น 2018–2026
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ วิธีรันบนเครื่อง

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Disclaimer

ผลพยากรณ์นี้เป็นเพียงการประมาณการทางสถิติจากข้อมูลในอดีต  
ไม่ใช่คำแนะนำด้านการลงทุน ใช้เพื่อการศึกษาเท่านั้น

# 📈 Stock Price Prediction App

พยากรณ์ราคาปิดหุ้นวันถัดไปด้วย Random Forest Regressor  
ครอบคลุมหุ้น 5 ตัว: AAPL · MSFT · AMZN · JPM · KO  
ข้อมูลช่วงปี 2014–2023

## 🚀 Demo
> วางลิงก์ Streamlit Cloud ที่นี่หลัง deploy

## 📁 โครงสร้างไฟล์
```
├── app.py                  # Streamlit app หลัก
├── requirements.txt        # Dependencies
├── StockPriceDataset.csv   # ชุดข้อมูลหุ้น
├── notebook/
│   └── untitled11-8.py     # โค้ด Colab ต้นฉบับ
└── README.md
```

## ⚙️ วิธีรันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Features ที่ใช้ (13 Features)
| Feature | คำอธิบาย |
|---------|----------|
| Close, Open, High, Low | ราคาของวันนั้น |
| Volume | ปริมาณซื้อขาย |
| Returns | % เปลี่ยนแปลงรายวัน |
| MA5, MA20, MA50 | Moving Averages |
| Volatility | ความผันผวน 20 วัน |
| RSI | Relative Strength Index 14 วัน |
| Price_Range | High - Low |
| Volume_Ratio | Volume เทียบ MA20 |

## 📊 โมเดลและผลลัพธ์
- **Algorithm**: Random Forest Regressor (200 trees)
- **Target**: ราคาปิดวันถัดไป
- **Split**: Time-based 80/20

## ⚠️ Disclaimer
ผลพยากรณ์เป็นเพียงการประมาณการทางสถิติ ไม่ใช่คำแนะนำด้านการลงทุน

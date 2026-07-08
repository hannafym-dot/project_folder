import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- ตั้งค่าหน้าเว็บ (โทนสีสามารถปรับได้ผ่าน CSS) ---
st.set_page_config(page_title="Risk Prediction Web App", page_icon="🔬", layout="wide")

# --- CSS ตกแต่งเพิ่มเติม ---
st.markdown("""
    <style>
    .main { background-color: #FAFAFA; }
    h1, h2, h3 { color: #2B3A67; } /* โทนน้ำเงินเข้ม */
    .stButton>button { background-color: #6A4C93; color: white; border-radius: 8px; } /* โทนม่วง */
    </style>
""", unsafe_allow_html=True)

# --- ฟังก์ชันทำนายแบบ Rule-based (Demo) ---
def rule_based_prediction(moisture, storage_day, temp, ph):
    # กฎสมมติสำหรับการประเมินเบื้องต้น
    if moisture < 20 and temp <= 25 and ph <= 5.0:
        return "Low", np.random.randint(80, 95)
    elif moisture > 40 or temp >= 30 or ph >= 6.5:
        return "High", np.random.randint(85, 98)
    else:
        return "Medium", np.random.randint(70, 89)

# --- Sidebar Menu ---
st.sidebar.title("🔬 เมนูนำทาง")
menu = st.sidebar.radio("เลือกหน้าต่างที่ต้องการ", 
                        ["Home", "Risk Prediction", "Dataset", "Model Info", "About Project"])

# ==========================================
# 1. หน้า Home
# ==========================================
if menu == "Home":
    st.title("เว็บไซต์ XGBoost สำหรับวิเคราะห์ความเสี่ยงจากข้อมูลแผ่นดูดซับความชื้นแบคทีเรียลเซลลูโลสผสมแอนโทไซยานิน")
    st.markdown("---")
    st.write("**ยินดีต้อนรับสู่ระบบประเมินความเสี่ยงเบื้องต้น** ระบบนี้ถูกออกแบบมาเพื่อวิเคราะห์สภาวะแวดล้อมที่มีผลต่อความเสี่ยงในการเสื่อมสภาพของอาหาร โดยอ้างอิงจากข้อมูลแผ่นดูดซับอัจฉริยะ")
    
    st.subheader("⚙️ หลักการทำงาน")
    st.write("1. **กรอกข้อมูล:** ผู้ใช้ป้อนค่าจากการทดลอง")
    st.write("2. **คำนวณความชื้น:** ระบบคำนวณอัตราการดูดซับความชื้นอัตโนมัติ")
    st.write("3. **XGBoost วิเคราะห์:** ส่งตัวแปรเข้าสู่โมเดล Machine Learning เพื่อประเมินผล")
    st.write("4. **แสดงระดับความเสี่ยง:** ประมวลผลออกมาเป็นระดับ ต่ำ / ปานกลาง / สูง")
    
    st.warning("⚠️ **คำเตือนทางวิทยาศาสตร์:** ระบบนี้เป็นเพียงเครื่องมือประเมินความเสี่ยงเบื้องต้นเพื่อเฝ้าระวังสภาวะที่เหมาะสมต่อการเจริญเติบโตของจุลินทรีย์ ไม่ใช่การยืนยันการพบเชื้อรา *Aspergillus flavus* หรือ Aflatoxin โดยตรง")

# ==========================================
# 2. หน้า Risk Prediction
# ==========================================
elif menu == "Risk Prediction":
    st.title("📊 ระบบวิเคราะห์ความเสี่ยง (Risk Prediction)")
    st.markdown("กรุณากรอกข้อมูลจากการทดลองเพื่อประเมินความเสี่ยง")
    
    col1, col2 = st.columns(2)
    with col1:
        initial_weight = st.number_input("น้ำหนักเริ่มต้นของแผ่นก่อนทดลอง (กรัม)", min_value=0.01, value=1.00, step=0.01)
        final_weight = st.number_input("น้ำหนักหลังเสร็จสิ้นการทดลอง (กรัม)", min_value=0.00, value=1.20, step=0.01)
        storage_day = st.number_input("ระยะเวลาเก็บรักษา/ระยะเวลาทดลอง (วัน)", min_value=1, value=7, step=1)
    with col2:
        temperature = st.number_input("อุณหภูมิ (°C)", min_value=0.0, value=25.0, step=0.5)
        ph = st.number_input("ค่า pH", min_value=0.0, max_value=14.0, value=5.5, step=0.1)

    if st.button("วิเคราะห์ความเสี่ยง"):
        # คำนวณ Moisture Absorption (%)
        moisture_absorption = ((final_weight - initial_weight) / initial_weight) * 100
        
        st.markdown("### ผลการคำนวณ")
        st.info(f"**Moisture Absorption:** {moisture_absorption:.2f} %")
        
        # เตรียมข้อมูลสำหรับโมเดล
        features = pd.DataFrame([[moisture_absorption, storage_day, temperature, ph]], 
                                columns=['moisture_absorption_percent', 'storage_day', 'temperature', 'ph'])
        
        risk_level = ""
        confidence = 0
        is_demo = False
        
        # พยายามโหลดโมเดล XGBoost
        model_path = "model.pkl"
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # XGBoost คืนค่า 0,1,2 (สมมติให้ 0=Low, 1=Medium, 2=High)
                pred_encoded = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = max(proba) * 100
                
                mapping = {0: "Low", 1: "Medium", 2: "High"}
                risk_level = mapping[pred_encoded]
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
                risk_level, confidence = rule_based_prediction(moisture_absorption, storage_day, temperature, ph)
                is_demo = True
        else:
            risk_level, confidence = rule_based_prediction(moisture_absorption, storage_day, temperature, ph)
            is_demo = True
            
        st.markdown("### 🎯 ผลการประเมินความเสี่ยงเบื้องต้น")
        if is_demo:
            st.caption("*หมายเหตุ: ผลลัพธ์นี้มาจากระบบ Rule-based Demo เนื่องจากยังไม่มีไฟล์โมเดลที่เทรนจากข้อมูลจริง*")
            
        if risk_level == "Low":
            st.success(f"🟢 **ระดับความเสี่ยง: ต่ำ (Low)** | ความมั่นใจ: {confidence:.2f}%")
            st.write("**คำอธิบาย:** สภาวะแวดล้อมยังอยู่ในเกณฑ์ที่ปลอดภัย ความชื้นและอุณหภูมิไม่เอื้อต่อการเสื่อมสภาพอย่างรวดเร็ว")
            st.write("**คำแนะนำ:** สามารถจัดเก็บในสภาวะนี้ต่อไปได้ แต่ควรตรวจสอบสม่ำเสมอ")
        elif risk_level == "Medium":
            st.warning(f"🟡 **ระดับความเสี่ยง: ปานกลาง (Medium)** | ความมั่นใจ: {confidence:.2f}%")
            st.write("**คำอธิบาย:** สภาวะแวดล้อมเริ่มมีการเปลี่ยนแปลง ความชื้นหรือค่า pH อยู่ในระดับที่ควรเฝ้าระวัง")
            st.write("**คำแนะนำ:** ควรตรวจสอบสภาพอาหารและแผ่นดูดซับอย่างใกล้ชิด อาจพิจารณาปรับลดอุณหภูมิ")
        else:
            st.error(f"🔴 **ระดับความเสี่ยง: สูง (High)** | ความมั่นใจ: {confidence:.2f}%")
            st.write("**คำอธิบาย:** สภาวะแวดล้อมมีความชื้นสูง อุณหภูมิหรือค่า pH เอื้อต่อการเจริญเติบโตของจุลินทรีย์")
            st.write("**คำแนะนำ:** ควรแยกอาหารหรือผลิตภัณฑ์ออกทันที และตรวจสอบการเน่าเสีย ไม่แนะนำให้บริโภค")

# ==========================================
# 3. หน้า Dataset
# ==========================================
elif menu == "Dataset":
    st.title("📂 ข้อมูลตัวอย่าง (Dataset)")
    st.write("ตัวอย่างข้อมูลที่ใช้สำหรับการเทรนโมเดล XGBoost")
    
    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 ดาวน์โหลดไฟล์ dataset.csv",
            data=csv,
            file_name='dataset.csv',
            mime='text/csv',
        )
    else:
        st.warning("ไม่พบไฟล์ dataset.csv ในระบบ")
        
    st.markdown("### คำอธิบายคอลัมน์")
    st.write("- **initial_weight:** น้ำหนักแผ่นก่อนทดลอง (g)")
    st.write("- **final_weight:** น้ำหนักแผ่นหลังทดลอง (g)")
    st.write("- **moisture_absorption_percent:** เปอร์เซ็นต์ความชื้นที่ดูดซับได้")
    st.write("- **storage_day:** จำนวนวันที่เก็บรักษา")
    st.write("- **temperature:** อุณหภูมิ (°C)")
    st.write("- **ph:** ค่า pH (ระดับความเป็นกรด-ด่าง)")
    st.write("- **risk_level:** ระดับความเสี่ยง (Low, Medium, High)")

# ==========================================
# 4. หน้า Model Info
# ==========================================
elif menu == "Model Info":
    st.title("🧠 ข้อมูลโมเดล (Model Info)")
    
    st.subheader("XGBoost คืออะไร?")
    st.write("XGBoost (Extreme Gradient Boosting) เป็นอัลกอริทึม Machine Learning ประเภท Tree-based ที่ได้รับความนิยมอย่างมาก มีจุดเด่นด้านความแม่นยำสูงและการประมวลผลที่รวดเร็ว เหมาะสำหรับการทำนายความเสี่ยงจากข้อมูลตัวเลข")
    
    st.subheader("ตัวแปรที่ใช้ (Features)")
    st.write("โมเดลใช้ 4 ตัวแปรหลักในการวิเคราะห์:")
    st.write("1. ความชื้นที่ดูดซับได้ (%)")
    st.write("2. ระยะเวลาการเก็บรักษา (วัน)")
    st.write("3. อุณหภูมิ (°C)")
    st.write("4. ค่า pH")
    
    st.subheader("ผลลัพธ์ (Output Classes)")
    st.write("- **Low:** สภาวะปลอดภัย")
    st.write("- **Medium:** สภาวะเฝ้าระวัง")
    st.write("- **High:** สภาวะเสี่ยงสูง")
    
    st.info("💡 **สถานะปัจจุบัน:** หากยังไม่มีข้อมูล Dataset จริง ระบบจะใช้ Rule-based Demo เพื่อแสดงผลลัพธ์การทำงานไปก่อน เมื่อเก็บข้อมูลจริงได้เพียงพอแล้ว สามารถรันไฟล์ `train_model.py` เพื่อสร้างไฟล์ `model.pkl` และให้เว็บไซต์สลับไปใช้ XGBoost อัตโนมัติ")

# ==========================================
# 5. หน้า About Project
# ==========================================
elif menu == "About Project":
    st.title("🌱 เกี่ยวกับโครงงาน (About Project)")
    
    st.subheader("แผ่นแบคทีเรียลเซลลูโลส (Bacterial Cellulose)")
    st.write("เป็นเส้นใยเซลลูโลสที่สร้างจากแบคทีเรีย มีคุณสมบัติดูดซับน้ำได้ดีเยี่ยม มีความเหนียว และเข้ากันได้ทางชีวภาพ (Biocompatibility) จึงเหมาะนำมาทำเป็นแผ่นดูดซับในบรรจุภัณฑ์อาหาร")
    
    st.subheader("แอนโทไซยานินจากดอกอัญชัน (Anthocyanin)")
    st.write("เป็นสารสีธรรมชาติที่เปลี่ยนสีได้ตามค่า pH การนำมาผสมในแผ่นดูดซับ ช่วยให้เราสังเกตความเปลี่ยนแปลงของความเป็นกรด-ด่างในบรรจุภัณฑ์ได้อย่างง่ายดาย ซึ่งบ่งบอกถึงสภาวะการเน่าเสีย")
    
    st.subheader("ความเชื่อมโยงกับความชื้น ค่า pH และความเสี่ยง")
    st.write("เมื่ออาหารเริ่มเสื่อมสภาพ จุลินทรีย์จะเจริญเติบโต ทำให้เกิดความชื้นและสร้างกรด (ทำให้อุณหภูมิและ pH เปลี่ยนไป) ระบบของเราจึงใช้ตัวแปรเหล่านี้มาประเมินความเสี่ยงเบื้องต้น (Preliminary Risk Assessment)")
    
    st.subheader("ประโยชน์ต่อความปลอดภัยทางอาหาร")
    st.write("เครื่องมือนี้ช่วยให้เกษตรกร ผู้ประกอบการ และผู้บริโภค สามารถเฝ้าระวังความเสี่ยงการเกิดเชื้อราและสารพิษ เช่น Aflatoxin ได้ในเบื้องต้น ลดความสูญเสียทางเศรษฐกิจและปัญหาด้านสุขภาพ")
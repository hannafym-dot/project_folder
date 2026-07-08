# XGBoost Risk Prediction App 🔬

เว็บไซต์สำหรับการประเมินความเสี่ยงเบื้องต้น จากข้อมูลแผ่นดูดซับความชื้นแบคทีเรียลเซลลูโลสผสมแอนโทไซยานิน

## สูตรการคำนวณที่ใช้ในระบบ
การคำนวณอัตราการดูดซับความชื้น (Moisture Absorption):

$$Moisture Absorption (\%) = \left( \frac{Final Weight - Initial Weight}{Initial Weight} \right) \times 100$$

## วิธีการติดตั้ง (Installation)
1. ตรวจสอบว่าติดตั้ง Python 3.8 ขึ้นไปในเครื่องแล้ว
2. เปิด Terminal หรือ Command Prompt
3. ติดตั้งไลบรารีที่จำเป็นด้วยคำสั่ง:
   ```bash
   pip install -r requirements.txt
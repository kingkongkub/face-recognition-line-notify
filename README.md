# Face Recognition with LINE Notify
ระบบสแกนใบหน้าและแจ้งเตือนผ่าน LINE Notify

## วิธีการใช้งาน
1. ติดตั้ง library ด้วยคำสั่ง `pip install -r requirements.txt`
2. คัดลอกไฟล์ `config.example.yaml` และเปลี่ยนชื่อเป็น `config.yaml`
3. ใส่ LINE Notify Token ของคุณลงในไฟล์ `config.yaml`
4. นำรูปภาพใบหน้าไปใส่ในโฟลเดอร์ `images` และอัปเดตชื่อใน `config.yaml`
5. รันโปรแกรมด้วยคำสั่ง `python FaceRecognition.py`

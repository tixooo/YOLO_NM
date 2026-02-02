Зависимости:

pip install torch torchvision opencv-python
git clone https://github.com/ultralytics/yolov5
pip install ultralytics --break-system-packages
cd yolov5
pip install -r requirements.txt

Слагаме Video.mp4 в главната директория при файловете

Input video - https://drive.google.com/file/d/1mwMbmHMNwb044Idu4vG145kfWNkpUDsc/view?usp=sharing
Output video - https://drive.google.com/file/d/15x8JL-3dsYqaCpjvCtcUb0l7FARE6l8F/view?usp=sharing

използваме:

python yolo_integration.py --source video.mp4


или

Използваме Python директно с конзолен дебъг за обратна връзка:

python3 << 'EOF'
from ultralytics import YOLO

print("Зареждам модел...")
model = YOLO('yolov8n.pt')

print("Обработвам видеото...")
results = model('video.mp4', save=True, conf=0.25)

print("\n✓ Готово!")
print("Резултатът е в: runs/detect/predict/")
EOF


Примери:

Webcam real-time detection
python detect.py --source 0 --weights yolov5s.pt

Видео с по-висок confidence threshold
python detect.py --source video.mp4 --weights yolov5m.pt --conf 0.5

Само човек и кола
python detect.py --source video.mp4 --weights yolov5s.pt --classes 0 2

На GPU
python detect.py --source video.mp4 --weights yolov5s.pt --device 0

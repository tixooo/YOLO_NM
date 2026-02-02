Зависимости:

pip install torch torchvision opencv-python
git clone https://github.com/ultralytics/yolov5
pip install ultralytics --break-system-packages
cd yolov5
pip install -r requirements.txt

Слагаме Video.mp4 в главната директория при файловете

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

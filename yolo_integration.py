"""
Пример за интеграция на истински YOLOv8 модел
Този скрипт показва как да използваш реален YOLO за object detection
"""

# За да работи този код трябва да инсталираш:
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path


def detect_with_yolov8(video_path, output_path, model_name='yolov8n.pt', conf=0.25):
    """
    Обработва видео с YOLOv8 модел
    
    Parameters:
    -----------
    video_path : str
        Път към входното видео
    output_path : str
        Път за изходното видео
    model_name : str
        Име на модела (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    conf : float
        Минимална увереност за detection
    """
    
    print(f"Зареждам YOLOv8 модел: {model_name}")
    
    # Зареждаме модела - автоматично се тегли ако го няма
    model = YOLO(model_name)
    
    # Отваряме видеото
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Грешка: Не мога да отворя {video_path}")
        return False
    
    # Параметри на видеото
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nВидео информация:")
    print(f"  Резолюция: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Кадри: {total_frames}")
    
    # Създаваме VideoWriter за изхода
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("\nОбработвам видеото...")
    
    # Обработваме всеки кадър
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO inference на текущия кадър
        # verbose=False за да не се принтят detections за всеки кадър
        results = model(frame, conf=conf, verbose=False)
        
        # Рисуваме детекциите върху кадъра
        # plot() метода автоматично добавя boxes и labels
        annotated_frame = results[0].plot()
        
        # Записваме обработения кадър
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Прогрес: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    # Затваряме всичко
    cap.release()
    out.release()
    
    print(f"\nГотово! Обработени {frame_count} кадъра")
    print(f"Резултат: {output_path}")
    
    return True


def detect_with_yolov8_advanced(video_path, output_path, model_name='yolov8n.pt', conf=0.25):
    """
    По-напреднал вариант с повече контрол над детекциите
    """
    
    print(f"Зареждам модел: {model_name}")
    model = YOLO(model_name)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Статистика за различните класове
    class_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        results = model(frame, conf=conf, verbose=False)
        
        # Обхождаме детекциите
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Вземаме информацията за всеки detection
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Име на класа
                class_name = model.names[cls]
                
                # Броим колко пъти се среща всеки клас
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                # Рисуваме bounding box
                color = (0, 255, 0)  # зелен
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Добавяме текст
                label = f"{class_name} {conf_score:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Принтираме статистика
    print("\n" + "="*50)
    print("Статистика за разпознати обекти:")
    print("="*50)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    print("="*50)
    
    return True


# Пример за streaming от webcam в реално време
def webcam_detection(model_name='yolov8n.pt', conf=0.25):
    """
    Real-time detection от webcam
    Натисни 'q' за да спреш
    """
    
    print("Стартирам webcam detection...")
    print("Натисни 'q' за да спреш")
    
    model = YOLO(model_name)
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detection
        results = model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()
        
        # Показваме кадъра
        cv2.imshow('YOLOv8 Webcam', annotated_frame)
        
        # Излизаме при натискане на 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# TODO: добави object tracking за да следим обектите между кадрите
# TODO: имплементирай zone detection - алерт когато обект влезе в определена зона
# TODO: добави heatmap за да видим къде се появяват най-често обектите
# TODO: интегрирай с database за да записваш всички detections


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--source', type=str, default='test_video.mp4',
                       help='Входно видео')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Изходно видео')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 модел (n, s, m, l, x)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--webcam', action='store_true',
                       help='Използвай webcam')
    parser.add_argument('--advanced', action='store_true',
                       help='Използвай advanced режим със статистика')
    
    args = parser.parse_args()
    
    # Webcam режим
    if args.webcam:
        webcam_detection(args.model, args.conf)
        return
    
    # Проверка дали файла съществува
    if not Path(args.source).exists():
        print(f"Грешка: Файлът {args.source} не съществува!")
        return
    
    # Избираме метода за обработка
    if args.advanced:
        success = detect_with_yolov8_advanced(
            args.source, args.output, args.model, args.conf
        )
    else:
        success = detect_with_yolov8(
            args.source, args.output, args.model, args.conf
        )
    
    if success:
        print("\n✓ Detection завърши успешно!")


if __name__ == '__main__':
    main()


"""
ПРИМЕРИ ЗА ИЗПОЛЗВАНЕ:
======================

1. Основна употреба:
   python yolo_integration.py --source video.mp4 --output result.mp4

2. С по-голям модел:
   python yolo_integration.py --source video.mp4 --model yolov8m.pt

3. По-висок confidence:
   python yolo_integration.py --source video.mp4 --conf 0.5

4. Advanced режим със статистика:
   python yolo_integration.py --source video.mp4 --advanced

5. Webcam detection:
   python yolo_integration.py --webcam

6. Webcam с по-малък модел:
   python yolo_integration.py --webcam --model yolov8n.pt


НАЛИЧНИ МОДЕЛИ:
===============

yolov8n.pt - Nano (най-бърз, 6.2 MB)
yolov8s.pt - Small (бърз, 22.5 MB)
yolov8m.pt - Medium (балансиран, 49.7 MB)
yolov8l.pt - Large (по-точен, 83.7 MB)
yolov8x.pt - Extra Large (най-точен, 130.5 MB)


ДОПЪЛНИТЕЛНИ ВЪЗМОЖНОСТИ НА YOLOv8:
====================================

1. Segmentation (pixel-level detection):
   model = YOLO('yolov8n-seg.pt')

2. Pose estimation:
   model = YOLO('yolov8n-pose.pt')

3. Classification:
   model = YOLO('yolov8n-cls.pt')

4. Tracking:
   results = model.track(frame, persist=True)

5. Custom training:
   model = YOLO('yolov8n.pt')
   model.train(data='custom_data.yaml', epochs=100)
"""

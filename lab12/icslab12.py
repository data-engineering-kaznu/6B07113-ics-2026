import cv2
import csv

# Чтение входного файла
with open('input12.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = next(reader)
    video_path = data['video_path']

cap = cv2.VideoCapture(video_path)

tracker = cv2.TrackerKCF_create()

ret, frame = cap.read()
if not ret:
    print("Ошибка загрузки видео")
    exit()

# Выбор объекта
bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)

frame_count = 0
results = []

# Отслеживание
while True:
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        results.append([frame_count, x, y, w, h])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Tracking", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Запись результатов
with open('output12.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'x', 'y', 'width', 'height'])
    writer.writerows(results)

print("Результаты сохранены в output12.csv")
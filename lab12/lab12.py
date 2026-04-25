import cv2

cap = cv2.VideoCapture(0)
tracker = None
tracking = False
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read() # ytw
    if not ret: # nd
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # rgb bmp
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if not tracking:
        frame_diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1] # > w

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(contour) # pryamoug
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True
            break

    else: # alr t
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            tracking = False
    prev_gray = gray
    cv2.imshow("Object Tracking (OpenCV)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
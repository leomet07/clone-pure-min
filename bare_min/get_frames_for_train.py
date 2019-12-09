import cv2

cap = cv2.VideoCapture("rtsp://user2:user#55@46.52.182.186:37778/cam/realmonitor?channel=2&subtype=0")
write = True
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (720 , 480))
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if write:
        output_movie = cv2.VideoWriter('output_for_train.mp4', fourcc, fps, (w, h))
    cv2.imshow("window",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

import cv2


cap = cv2.VideoCapture(0) # 0 is the index of camera you want to use
cap.set(cv2.CAP_PROP_FPS, 60) # set 60 fps


face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read() # ret - True if the frame was read successfuly
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
    minNeighbors=5)

    for (x, y, w, h) in faces:
        # draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # show the frame in window
    cv2.imshow("frame", frame)


    # press q to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



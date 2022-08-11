import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for(x, y, w, h) in faces:
      cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2)

    font = cv2.FONT_ITALIC

    cv2.putText(frame, result['dominant_emotion'], (50,50), font, 2, (0,0,255), 2, cv2.LINE_4)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
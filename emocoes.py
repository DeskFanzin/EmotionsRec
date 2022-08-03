import cv2
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    cv2.imshow("Video", frame)
    
    #codigo do indiano
    result = DeepFace.analyze(frame,actions=['emotion'], enforce_detection=False)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for(x, y, w, h) in faces:
      cv2.rectangle(frame, (x,y),(x+w, y+h),(0,255,0),2) 

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (255,0,0), 2, cv2.LINE_4)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
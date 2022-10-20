import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():

        ret, frame = cap.read()
        height = int(cap.get(4))

        gsimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        result = human_cascade.detectMultiScale(gsimg)
        for (x,y,w,h) in result:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        frame.flags.writeable = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame)

        frame.flags.writeable = True
        font = cv2.FONT_HERSHEY_COMPLEX
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.detections:
          for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
            frame = cv2.putText(frame,'HUMANS DETECTED INITIATING TERMINATION....',(10,height-10),font,1,(0,255,0),5,cv2.LINE_AA)


        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
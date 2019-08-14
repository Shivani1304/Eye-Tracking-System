import numpy as np
import cv2
from imutils import face_utils
import pyautogui as pag
import imutils
import dlib


EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 3
WINK_AR_DIFF_THRESH = 0.05
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10

blink_counter = 0
wink_counter = 0
ANCHOR_POINT= (0,0)
active_input = False
eye_blink = False
left_wink = False   
right_wink = False
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
   
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0 
cap = cv2.VideoCapture (0)
#eye_path = "haarcascade_eye.xml"
shape_predictor = "shape_predictor_68_face_landmarks.dat"
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(shape_predictor)
#eye_cascade = cv2.CascadeClassifier(eye_path)

(Lstart, Lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(Rstart, Rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(Nstart, Nend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

#to detect eye
def eye_aspect_ratio(eye):
    X = np.linalg.norm(eye[1] - eye[5])
    Y = np.linalg.norm(eye[2] - eye[4])
    Z = np.linalg.norm(eye[0] - eye[3])
    ear = (X + Y) / (2.0 * Z)
    return ear

def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return 'right'
    elif nx < x - multiple * w:
        return 'left'

    if ny > y + multiple * h:
        return 'down'
    elif ny < y - multiple * h:
        return 'up'

    return '-'

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize( frame,(0,0), fx=1, fy=1)
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
    rects = detect(gray, 0)

    
    
    # Loop over the face detections
    
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame",frame)
        ch = cv2.waitKey(1) & 0xFF
        continue

    shape = predict(gray, rect)
    shape = face_utils.shape_to_np(shape)

    LeftEye = shape[Lstart:Lend]
    RightEye = shape[Rstart:Rend]
    nose = shape[Nstart:Nend]

    temp = LeftEye
    LeftEye = RightEye
    RightEye = temp
    #Average Aspect ratio

    LeftEAR = eye_aspect_ratio(LeftEye)
    RightEAR = eye_aspect_ratio(RightEye)
    ear = (LeftEAR + RightEAR) / 2.0
    differance = np.abs(LeftEAR - RightEAR)
    nose_point = (nose[3, 0], nose[3, 1])
    
    leftEyeHull = cv2.convexHull(LeftEye)
    rightEyeHull = cv2.convexHull(RightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (255,0,0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (255,0,0), 1)

    for (x, y) in np.concatenate((nose,LeftEye, RightEye), axis=0):
        cv2.circle(frame, (x, y), 2, (0,255,0), -1)

    #mouse events
    #if differance > WINK_AR_DIFF_THRESH:
        if ear <= EYE_AR_THRESH:
            blink_counter += 1
            if blink_counter > EYE_AR_CONSEC_FRAMES:
                active_input = not active_input
                blink_counter = 0
                ANCHOR_POINT = nose_point
                
        if LeftEAR < EYE_AR_THRESH:
            COUNTER_LEFT += 1  
        else:
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:  
                pag.click(button = 'left')
                COUNTER_LEFT = 0  
   
        if RightEAR < EYE_AR_THRESH:  
           COUNTER_RIGHT += 1
        else:
           if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:  
               pag.click(button = 'right') 
               COUNTER_RIGHT = 0  
   
    #cv2.putText(frame, "Wink Left : {}".format(TOTAL_LEFT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
    #cv2.putText(frame, "Wink Right: {}".format(TOTAL_RIGHT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
        
    
    if active_input:
        cv2.putText(frame,"Mouse event Activated", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        dir = direction(nose_point, ANCHOR_POINT, w, h)
        drag = 16

        if dir == 'right':
            pag.moveRel(drag,0)

        elif dir == 'left':
            pag.moveRel(-drag,0)

        elif dir == 'up':
            pag.moveRel(0,-drag)

        elif dir == 'down':
            pag.moveRel (0, drag)
            EYE_AR_CONSEC_FRAMES
    cv2.imshow("Frame",frame)
    ch = cv2.waitKey(1)
                    
    if ch & 0xFF == ord ('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

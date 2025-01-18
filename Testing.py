import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
model_select=pickle.load(open('./model.pickle','rb'))
model1=model_select['model1']
model2=model_select['model2']
model3=model_select['model3']
model4=model_select['model4']

label_list= {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'}

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles

hand=mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap=cv2.VideoCapture(0)

while True:
    data_aux=[]
    x_=[]
    y_=[]
    ret,frame= cap.read()
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hand.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(), 
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y

                data_aux.append(x-min(x_))
                data_aux.append(y-min(y_))

        prediction = model1.predict([np.asarray(data_aux)])        
        predicted_letter=label_list[int(prediction[0])]
        print(predicted_letter)
    cv2.imshow('Frame',frame)
    cv2.waitKey(1)
        
cap.release()
cv2.destroyAllWindows()
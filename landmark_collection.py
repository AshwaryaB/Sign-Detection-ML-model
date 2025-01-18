import pickle
import os
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2


mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
Hand = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


Data_dir ='./data'

data=[]
label=[]

for dir_ in os.listdir(Data_dir):
    for img_no in os.listdir(os.path.join(Data_dir,dir_)):
        data_aux=[]

        x_=[]
        y_=[]
        img_no=cv2.imread(os.path.join(Data_dir,dir_,img_no))
        img_new=cv2.cvtColor(img_no,cv2.COLOR_BGR2RGB)
        results=Hand.process(img_new)
        
        if results.multi_hand_landmarks:
         
            for landmrk in results.multi_hand_landmarks:
                for i in range(len(landmrk.landmark)):
                    x=landmrk.landmark[i].x
                    y=landmrk.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(landmrk.landmark)):
                    x=landmrk.landmark[i].x
                    y=landmrk.landmark[i].y

                    data_aux.append(x - min(x_))
                    data_aux.append(x - min(y_))
            label.append(dir_)
            data.append(data_aux)

 
f=open('data.pkl','wb')
pickle.dump({'data':data,'label':label},f)
f.close()






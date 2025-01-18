import cv2
import os

data_dir='./data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

classes =7
images =500
cap= cv2.VideoCapture(0)

for j in range(classes):
   if not os.path.exists(os.path.join(data_dir,str(j))):
       os.makedirs(os.path.join(data_dir,str(j)))
    
   print('collecting data for class{}'.format(j))
    
   done = False
   while(True):
       ret,frame=cap.read()
       cv2.putText(frame,'Press spacebar to capture',(100,50),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0.255),3,cv2.LINE_AA)
       cv2.imshow('frame',frame)
       if cv2.waitKey(25)==ord(' '):
           break
   counter=0
   while counter<images:
       ret,frame=cap.read()
       cv2.imshow('frame',frame)
       cv2.waitKey(25)
       cv2.imwrite(os.path.join(data_dir,str(j),"{}.jpg".format(counter)),frame)
       counter+=1

cap.release
cv2.destroyAllWindows


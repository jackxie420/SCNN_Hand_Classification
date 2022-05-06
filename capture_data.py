import cv2 
import numpy as np
import time
import mediapipe as mp
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm_notebook
from sklearn.utils import shuffle
from scipy.special import softmax
import csv

vid = cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mpDraw=mp.solutions.drawing_utils

letters=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

input_label='a4'

#number_index=letters.index(input_label)
#print(number_index)


myFile = open('/Users/jx/Desktop/Col/ISEF/isef_new_program/my_dataset/disease_dataset2/'+input_label+'.csv', 'w')

file_list=[]

time_start=time.time()

while(time.time()-time_start<60):
    
    
    label, frame = vid.read()
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(RGBframe)
    
    #lm_list=[number_index]
    
    
    frame_h,frame_w,_=frame.shape
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            lm_list=[]
            for idx, cur_lm in enumerate(landmarks.landmark):  
                lm_list.append(cur_lm.x) 
                lm_list.append(cur_lm.y) 
                lm_list.append(cur_lm.z) 
            file_list.append(lm_list)
            
            mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
                 
        
    cv2.imshow("Image", frame)                  
        
    cv2.waitKey(1)



save_file_list=np.array(file_list).reshape(-1,63)
#.reshape(-1,63)

with myFile:
    writer = csv.writer(myFile)
    writer.writerows(save_file_list) 

print(len(file_list))

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



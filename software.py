import cv2 
import numpy as np
import time
import mediapipe as mp
from sqlalchemy import true
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from scipy.special import softmax
import csv
import tensorflow as tf

model_name="ASL_Fingerspelling_Model_parameters.h5"

new_model = tf.keras.models.load_model(model_name)

letters=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

#letters=["Healthy", "Index Finger", "Middle Finger", "Ring Finger", "Little Finger"]

vid = cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mpDraw=mp.solutions.drawing_utils

while(True):
    
    
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
            input=np.array(lm_list).reshape(-1,63,1)

            output=new_model.predict(input)

            pred_max=np.argmax(output)
            cv2.putText(frame,str(letters[pred_max]), (100,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        
            
            mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
                 
        
    cv2.imshow("Image", frame)                  
        
    cv2.waitKey(1)


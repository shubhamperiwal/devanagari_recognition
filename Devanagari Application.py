#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2  #Open CV - Computer Vision
from keras.models import load_model 
import numpy as np
from collections import deque #Queue (Stack and Queue)


# In[2]:


model1 = load_model('model/devanagari_model_refined.h5')
print(model1)


# In[3]:


letter_count = {0: 'Check', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna',
               6: '01_ka', 7: '07_kha', 8: '08_ga', 9: '04_gha', 10: '10_kna',
               11: '11_ka', 12: '12_kha', 13: '13_ga', 14: '14_gha', 15: '15_kna',
               16: '16_ka', 17: '17_kha', 18: '18_ga', 19: '19_gha', 20: '20_kna',
               21: '21_ka', 22: '22_kha', 23: '23_ga', 24: '24_gha', 25: '25_kna',
               26: '26_ka', 27: '27_kha', 28: '28_ga', 29: '29_gha', 30: '30_kna',
               31: '31_ka', 32: '32_kha', 33: '33_ga', 34: '34_gha', 35: '35_kna',
               36: '36_gya', 37: '37_Check'}


# In[4]:


def keras_predict(model, image): #To predict data from webcam
    processed = keras_process_image(image)
    print("processed: "+str(processed.shape))
    pred_probab = model.predict(processed)[0]
    #Get class with max probability 
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

#Gets image, resize image to a 32x32 pixel size. Then convert to np array and reshape/ 
def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    # -1 to get last value. X,y is height,width and 1 is number of channels
    img=  np.reshape(img, (-1, image_x, image_y, 1))
    return img


# In[16]:


cap = cv2.VideoCapture(0)  #Get webcam feed
Lower_blue = np.array([110, 50, 50])  #BGR format. Capture the blue colours
Upper_blue = np.array([130, 255, 255])  #When camera captures value between 110-130, then it's probably a blue, so that's what we want
pred_class=0 
pts = deque(maxlen=512) #queue
blackboard = np.zeros((480, 640, 3), dtype=np.uint8) #Actual drawings are placed in this blackboard
digit = np.zeros((200, 200, 3), dtype=np.uint8) 

while (True):  #While we are actually capturing anything
    ret, img = cap.read() #Get image from webcam
    img = cv2.flip(img, 1)  #Flip image coz it'll probably be mirrored
    cv2.imshow('image', img)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Convert to HSV
    mask = cv2.inRange(imgHSV, Lower_blue, Upper_blue) #Mask between lower bound and upper bound blue. If anything between these 2, we want
    blur = cv2.medianBlur(mask, 15) #Blue image to get rid of edges
    blur = cv2.GaussianBlur(blur, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #ignore values beyond threshold
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1] #diff features on image beyond threshold
    center = None

    try:
        if(cnts is None):
            print("None")
        elif(len(cnts) >= 1): #if contour is present (Blue colour is present)
            print("CONT")
            contour = max(cnts, key=cv2.contourArea) #Take max of contour and calc area
            if(cv2.contourArea(contour) > 250):
                ((x,y), radius) = cv2.minEnclosingCircle(contour) #enclose contour in circle to ensure where contour is 
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)  #Creates circle around contour
                cv2.circle(img, center, 5, (0, 0, 255, -1))
                M = cv2.moments(contour)
                center = (int(M['m10'] / m['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center) #Insers center in a queue. 
                for i in range(1, len(pts)):
                    if(pts[i-1] is None or pts[i] is None):
                        continue
                    #Draw line in blackboard of black color and length 10
                    cv2.line(blackboard, pts[i-1], pts[i], (255, 255, 255), 10)
                    #Draw line in image of red color and size 5
                    cv2.line(img, pts[i-1], pts[i], (0, 0, 255), 5)
        elif(len(cnts) ==0): #Suppose we have removed blue colour on screen. No contours
            print("NO CONTOURS")
            if(len(pts) != []):
                #Convert to gray colour, blur, get threshold and get contour
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5,5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if(len(blackboard_cnts)) >=1:
                    #Get max of contour
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    #print area
                    print(cv2.contourArea(cnt))
                    #If area>2000, then it is noise. Might be blue color in background, etc
                    if cv2.contourArea(cnt) > 2000:
                        x,y,w,h = cv2.boundingRect(cnt) #Bound contour into rectangle
                        #Convert blackboard gray image to digit variable
                        digit = blackboard_gray[y:y + h, x:x + w]
                        #newImage = process_letter(digit)
                        pred_probab, pred_class = keras_predict(model1, digit)
                        print(pred_class, pred_probab)

            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Conv network: "+str(letter_count[pred_class]), (10, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
    except Exception as e:
        print('Except: '+str(e))
    k = cv2.waitKey(1)
    if  k & 0xFF == 27 or ret==False:
        break

cv2.destroyAllWindows()
cap.release()
                


# In[ ]:





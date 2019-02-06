#import potrebnih biblioteka
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import collections
import keras
import time
import pandas as pd
from scipy.spatial import distance
from scipy import ndimage
from functions import distance, pnt2line, length, vector
from Digit import Digit
# keRas
from sklearn import datasets
import matplotlib.pylab as pylab

suma = 0
digitArray= []

#Transformisati selektovani region na sliku dimenzija 28x28     **vezbe2**
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

# Elementi matrice image su vrednosti 0 ili 255.   Potrebno je skalirati sve elemente matrica na opseg od 0 do 1 **vezbe2**
def scale_to_range(image): 
    return image/255

#Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa **vezbe2**
def matrix_to_vector(image):
    return image.flatten()


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_bin = np.ndarray((height, width), dtype=np.uint8)
    _,image_bin = cv2.threshold(image_gs, 160, 255, cv2.THRESH_BINARY)
    return image_bin

#priprema za obucavanje cifre
def prepare(digit,x,y):
     image = img_bin[digit.y-7:y+7,digit.x-7:x+7]
     resized = resize_region(image)
     scale = scale_to_range(resized)
     matvec = matrix_to_vector(scale)
     retVal = np.reshape(matvec,(1, 784))

     return retVal


def exist(digit):
    x, y, w, h = digit
    for digit in digitArray:
        p1 = [digit.x+digit.w, digit.y+digit.h]
        p2 = [x+w, y+h]
        relation = length(vector(p1, p2))  #ovde se koristila metoda p2pdistance izmedju dve tacke
        if(relation < 19):
            return digit           
    return None


#Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
def select_roi(image_bin):
    
    _, contours, _ = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    regions = []

    for i in range(0, len(contours)):
        contour = contours[i] 
        x = cv2.boundingRect(contour)[0]
        y = cv2.boundingRect(contour)[1]
        w = cv2.boundingRect(contour)[2]
        h = cv2.boundingRect(contour)[3]  
        if h > 10:            #ako je visina veca od 10      
            existingDigit = exist(cv2.boundingRect(contour))
            if existingDigit is None:
                digit = Digit(x, y, w, h, False, None)
                global digitArray
                digitArray.append(digit)
            else:
                existingDigit.x = x
                existingDigit.y = y
                existingDigit.w = w
                existingDigit.h = h
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            region = image_bin[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            regions.append(resize_region(region))
    
    
    return regions

#detekcija plave linije
def detectLine(cap):
    _, frame = cap.read()

    grayBlue = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edgesBlue = cv2.Canny(grayBlue, 50, 150, apertureSize = 3)
    #blurBlue = cv2.GaussianBlur(edgesBlue,(7,7),1)
    linesBlue = cv2.HoughLinesP(edgesBlue, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    return linesBlue

#pronalazenje koordinata plave, uzimamo najvecu
def findLineCoordinates(linesBlue):
    Bx1 = 0
    By1 = 0
    Bx2 = 0
    By2 = 0
    for x1, y1, x2, y2 in linesBlue[0]:
        Bx1=x1
        By1=y1
        Bx2=x2
        By2=y2

    for line in linesBlue:
        for x1, y1, x2, y2 in line:
            if x1<Bx1:
                Bx1=x1
                By1=y1
            if x2>Bx2:
                By2=y2
                Bx2=x2

    return Bx1,Bx2,By1,By2

'''
#pokusaj da vrti kroz svih 10 videa
ann = keras.models.load_model("keras_mnist.h5")
i = 0
for i in range (3):  
    
    name = 'videos/video-'+str(i)+'.avi'

    linesBlue = detectLine(name)

    cap = cv2.VideoCapture(name)

    ret, frame1 = cap.read()

    grayBlue = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    edgesBlue = cv2.Canny(grayBlue, 50, 150, apertureSize = 3)
    linesBlue = cv2.HoughLinesP(edgesBlue, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    Bx1,Bx2,By1,By2 = findLineCoordinates(linesBlue)

    print ("Koordinate plave linije: (%d, %d) i (%d, %d)" %(Bx1,By1,Bx2,By2))

    cv2.line(frame1,(Bx1,By1),(Bx2,By2),(0,255,0),2)

    #prepoznavanje cifara
    while (1) :

        ret, frame = cap.read()


        if frame is not None:
      
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
            img_bin = image_bin(gray)

            slike = select_roi(img_bin)

            PointLine1=np.array([Bx1,By1])
            PointLine2=np.array([Bx2,By2])
    
            for digit in digitArray:
                x = digit.x + digit.w
                y = digit.y + digit.h
                counturePoint = np.array([x,y])
                dist,_,_ = pnt2line(counturePoint,PointLine1,PointLine2)
                if(dist <= 7):
                    if(digit.passed == False):
                        digit.passed = True
                        frameForTrain = np.array(prepare(digit,x,y), dtype=np.float32)
                        nmb = neuron.predict(frameForTrain)
                        predicted_number = np.argmax(nmb) 
                        cv2.putText(frame, str(predicted_number),(x-20,y-20), cv2.FONT_ITALIC, 2, (255,255,0))   
                        suma += predicted_number
        
            cv2.putText(frame,"Suma: "+str(suma),(20,50), cv2.FONT_ITALIC, 2, (0,255,0))
                
            cv2.imshow('video '+str(i), frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break        
    print (suma) 
    print ("Suma za video %d je: %d " %(i,suma))
    suma = 0   

    cap.release()
    cv2.destroyAllWindows()

#kraj toga
'''

#prvi frejm, detektujemo liniju
videoNum = 5
cap = cv2.VideoCapture("videos/video-"+str(videoNum)+".avi")

neuron = keras.models.load_model("keras_mnist.h5")

linesBlue = detectLine(cap)
Bx1,Bx2,By1,By2 = findLineCoordinates(linesBlue)
       
#print ("Koordinate plave linije: (%d, %d) i (%d, %d)" %(Bx1,By1,Bx2,By2))
    

#ovde pocinje glavna petlja dok se video ne zavrsi
while (1) :

    _, frame = cap.read()

    
    cv2.line(frame,(Bx1,By1),(Bx2,By2),(0,255,0),2)

    
    if frame is not None:

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
        img_bin = image_bin(img)

        slike = select_roi(img_bin)

        p1=np.array([Bx1,By1])
        p2=np.array([Bx2,By2])
    
        for digit in digitArray:
            x = digit.x + digit.w
            y = digit.y + digit.h
            point = np.array([x,y])
            dist,_,_ = pnt2line(point,p1,p2)
            if(dist <= 7):
                if(digit.passed == False):
                    digit.passed = True
                    frameForTrain = np.array(prepare(digit,x,y), dtype=np.float32)
                    nmb = neuron.predict(frameForTrain)
                    predicted_number = np.argmax(nmb) 
                    cv2.putText(frame, str(predicted_number),(x-20,y-20), cv2.FONT_ITALIC, 2, (255,255,0))   
                    suma += predicted_number
        
        cv2.putText(frame,"Suma: "+str(suma),(20,50), cv2.FONT_ITALIC, 2, (0,255,0))
           
        cv2.imshow('video ' + str(videoNum), frame)
        if cv2.waitKey(1) == 27: #key 27 je esc
            break
    else:
        break        
print (suma)    
cap.release()
cv2.destroyAllWindows()




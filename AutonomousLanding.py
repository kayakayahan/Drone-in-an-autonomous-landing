# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:17:19 2019

@author: Kayahan Kaya
"""
from scipy import ndimage
import numpy as np
import cv2
from random import choice
import random
from itertools import chain

img = cv2.imread('garden1.jpg')


kernel = np.array([
       [0, 0, 1, 0, 0], # elliptical Kernel 5x5
       [0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1],
       [0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0]], dtype="uint8")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

h,w,bpp = np.shape(img)

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

edge_low = np.array([ 0 , 0, 0])
edge_high = np.array([ 15 ,  15, 15])
mask = cv2.inRange(hsv_img,edge_low,edge_high)


ret,th = cv2.threshold(gray, 177, 255,cv2.THRESH_BINARY)


#erosion = cv2.erode(dilation,kernel,iterations = 2)
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


edges = cv2.Canny(closing,100,200)
dilation = cv2.dilate(edges, kernel, iterations=2)

erosion1 = cv2.erode(th,kernel,iterations=1)
dilation1 = cv2.dilate(erosion1,kernel,iterations=2)

contours, hierarchy = cv2.findContours(dilation.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours 
contours = sorted(contours, key = cv2.contourArea, reverse = False)
contours1 = sorted(contours, key = cv2.contourArea, reverse = False)

# Draw the contour 
img_copy = img.copy()
final = cv2.drawContours(img_copy, contours, contourIdx = -1, color = (255, 0, 0), thickness = 2)

draw = cv2.drawContours(img, contours, -1, (0,255,0), -1)

# The first order of the contours
c_0 = contours[0]

# Detect the convex contour
hull = cv2.convexHull(c_0)
img_copy = img.copy()
img_hull = cv2.drawContours(img_copy, contours = [hull], 
                            contourIdx = -0, 
                            color = (255, 0, 0), thickness = 1)

hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x,y,w,h = cv2.boundingRect(currentContour)
    if currentHierarchy[2] < 0:
        # these are the innermost child components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    elif currentHierarchy[3] < 0:
        # these are the outermost parent components
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)



  
def find_area(x,y):

    sequence1 = [xi for xi in range(550)]    
    sequence2 = [yi for yi in range(400)]       
    x = random.choice(sequence1)
    y = random.choice(sequence2)
    
    return(x,y)
    
    

lst_intensities = []

# For each list of contour points...
for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, contours, i, color=255, thickness=1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    lst_intensities.append(img[pts[0], pts[1]])

#print(sequence1)



color =(255,0,0)
h1 = 50
w1 = 50

k1 = np.array([])
i=0
j=0

a,b = find_area(x,y)
array1=[]
array2=[]
array3=[]
array4=[]
array5=[]
array6=[]
new_ar =[]
temp4 = [a,b]
array5.append(temp4)



for i in range(50):
    temp=[a,b+i]
    temp1 = [(a+i),b]
    temp2 = [a+i,b+i]
    temp3 = [a,b]
    
    array1.append(temp)
    array2.append(temp1)
    array3.append(temp2)
    array4.append(temp3)
    
 

mask = np.zeros(shape = img.shape, dtype = "uint8")

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    #print(r2[0])

    #print(r1)
    cv2.rectangle(img = mask, 
        pt1 = (x, y), 
        pt2 = (x + w, y + h), 
        color = (255, 255, 255), 
        thickness = -1)

image = cv2.bitwise_and(src1 = img, src2 = mask)
image2 = cv2.bitwise_or(image,mask)


#cv2.rectangle(image2, (a,b), (a+w1,b+h1), color, -1)  
image1 = cv2.bitwise_or(img,mask)

deneme=[]

deneme=np.concatenate((array1,array2,array3,array4))



npImg = np.asarray( dilation1 )
pixel = np.argwhere(npImg == [255])
numWhite = np.array(pixel)


result = np.all(numWhite == deneme[0] for numWhite in deneme)
print(list(result))
if np.any(deneme) in chain(*numWhite):
    a,b = find_area(x,y)
    
else:
    c = cv2.rectangle(dilation1, (a,b), (a+h1,b+w1), color, -1)
    
cv2.imshow("Rectangle1",dilation1)
#cv2.imshow("Contours",img)
cv2.imshow("Contours1",image)



cv2.waitKey(0)
cv2.destroyAllWindows()






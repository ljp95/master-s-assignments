import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

roi_defined = False

def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked,record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

''' video BEGIN '''
#cap = cv2.VideoCapture('Sequences/VOT-Ball.mp4')
#cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')
cap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
''' video END '''

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r, c), (r+h, c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking 
roi = clone[c:c+w, r:r+h]

#conversion to YUV space
yuv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2YUV)
#histogram of U and V components
roi_hist = cv2.calcHist([yuv_roi],[1,2],None,[255,255],[0,255,0,255])
#histogram values are normalised to ?
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria: either 10 iterations or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

''' functions BEGIN '''
def update_roi_hist(yuv,track_window,mask,roi_hist):
    (r,c,h,w) = track_window
    # set up the ROI for tracking in HSV space
    yuv_roi = yuv[c:c+w, r:r+h]
    #conversion to YUV space
    yuv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2YUV)
    #histogram of U and V components
    roi_hist = cv2.calcHist([yuv_roi],[1,2],None,[255,255],[0,255,0,255])
    #histogram values are normalised to [-0.5,0.5]
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist
    
''' functions END '''

cpt = 1

while(1):
    ret ,frame = cap.read()
    if ret == True:

        ''' mean-shift BEGIN '''

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	# Backproject the model histogram roi_hist onto the 
	# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
    
        dst = cv2.calcBackProject([yuv],[1,2],roi_hist,[0,255,0,255],1)

        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # apply the update to the histogram
#        roi_hist = update_roi_hist(yuv,track_window,mask,roi_hist)
        
        ''' mean-shift END '''
        # Draw a blue rectorientation on the current image
        x,y,h,w = track_window
        #frame_hsv
        frame_tracked = cv2.rectangle(frame, (x,y), (x+h,y+w), (255,0,0) ,2)
        ''' show BEGIN'''
        
        #sequence
        cv2.imshow('Sequence',frame_tracked)
        
        #weights
        cv2.imshow('Weights',dst)

        #yuv
        y,u,v = cv2.split(yuv)
        cv2.imshow('U component',u)
        cv2.imshow('V component',v)
        ''' show END '''
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
            cv2.imwrite('Weights_%04d.png'%cpt,dst)
            cv2.imwrite('U_%04d.png'%cpt,u)            
            cv2.imwrite('V_%04d.png'%cpt,v)            
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()





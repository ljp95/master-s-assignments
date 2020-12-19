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
#cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
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
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space : 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram : Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 

''' functions BEGIN '''

def local_or(frame,threshold):
    #going to grayscale
    img = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2GRAY)
    
    #gaussian blur to eliminate texture and noise
    #img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    
    #orientation and module of gradients
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
    mag, orientation = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

    module = np.sqrt(sobelx**2+sobely**2)
    
    #masking weak gradient module
    mask = module<threshold
    orientation[mask] = 0
    return orientation,mask,module

def create_R_table(track_window,orientation,alpha=10):
    (r,c,h,w) = track_window 
    
    #reference point at the center of the rectangle
    ref_pt = np.array([c+w//2,r+h//2])
    
    #init the R_table
    R_table = [[] for i in range(int(np.ceil(360/alpha)))]
    
    #filling it with vectors of points to ref_pt wrt gradient orientation
    for i in range(c,c+w,3): #step of 3 to speed up computation, don't need closed points especially of a contour
        for j in range(r,r+h,3):
            phi = orientation[i,j]
            if phi>=threshold:
                R_table[int(phi/alpha)].append(ref_pt-np.array([i,j]))  

    return R_table

def hough_transform(h,w,R_table,orientation,threshold,alpha=10):
    #init
    [n,m] = orientation.shape[0],orientation.shape[1]
    vote_mat = np.zeros((n,m))
    
    debut = time.time()
    #voting for the center
    vote_mat = np.zeros((n+2*h,m+2*w))
    index = np.where(orientation>=threshold)
    for i,j in zip(index[0],index[1]):
        phi = orientation[i,j]
        for a,b in R_table[int(phi/alpha)]:
            vote_mat[i+h+a,j+w+b] += 1
    vote_mat = vote_mat[h:n+h,w:m+w]
    
    #blurring to avoid unlucky positions
    vote_mat = cv2.GaussianBlur(vote_mat,(5,5),cv2.BORDER_DEFAULT)

    #retrieving the rectangle position
    y,x = np.argwhere(vote_mat == vote_mat.max())[0] - [w//2,h//2]
    
    print(time.time()-debut)
    return (x,y,h,w),vote_mat

    ''' functions END '''

#parameters
threshold = 50
alpha = 20
cpt = 1

#create the first R_table
orientation,mask,module = local_or(frame,threshold)
R_table = create_R_table(track_window,orientation,alpha)

while(1):
    ret ,frame = cap.read()
    if ret == True:
        
        ''' Hough transform + Mean-shift BEGIN '''
        #apply an update of the R_table
#        R_table = create_R_table(track_window,orientation,alpha)
        #compute thresholded orientation of gradient
        orientation,mask,module = local_or(frame,threshold)
        #apply hough transform
        _,vote_mat = hough_transform(h,w,R_table,orientation,threshold,alpha)
        ret, track_window = cv2.meanShift(vote_mat, track_window, term_crit)
        cv2.normalize(vote_mat,vote_mat,0,255,cv2.NORM_MINMAX)

        ''' Hough transform + Mean-shift END '''

        # Draw a blue rectorientation on the current image
        x,y,h,w = track_window
        #frame_hsv
        frame_tracked = cv2.rectangle(frame, (x,y), (x+h,y+w), (255,0,0) ,2)
        
        ''' show BEGIN '''
        #sequence
        cv2.imshow('Sequence',frame_tracked)
        
        #thresholded orientation
        img = np.zeros((orientation.shape[0],orientation.shape[1],3))
        img[:,:,0] = orientation
        img[:,:,1] = orientation
        img[:,:,2] = orientation
        img[mask] = [0,0,255]
        cv2.imshow('Orientation gradient thresholded',img)
        
        #vote matrix/weights
        cv2.imshow('Vote matrix/weights',vote_mat)
        ''' show END '''


        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
            cv2.imwrite('Grad_%04d.png'%cpt,img)
            cv2.imwrite('Vote_%04d.png'%cpt,vote_mat)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()


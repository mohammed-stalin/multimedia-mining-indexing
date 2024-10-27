import numpy as np
import cv2

def f(x):
    pass

def main():
    #read the two images 
    img1='../assets/mountain.jpg'
    img2='../assets/rose.jpg'

    a=cv2.imread(img1)
    b=cv2.imread(img2)
    #initialize alpha
    alpha=0

    c=np.uint8((1-alpha/100.)*a+(alpha/100.)*b)

    cv2.imshow('addition',c)

    #tracebare between 0 and 100 
    cv2.namedWindow('addition')  # Create a named window
    cv2.createTrackbar('Alpha:','addition',0,100,f)

    while True:
        if cv2.getTrackbarPos('Alpha:','addition') != alpha:
            # Get the current position of the trackbar
            alpha = cv2.getTrackbarPos('Alpha:','addition')

            c=np.uint8((1-alpha/100.)*a+(alpha/100.)+b)
            cv2.imshow('addition',c)
            
        if cv2.waitKey(0) & 0xFF==27:
            break
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()

import cv2  
input1='../assets/star.png'
input2='../assets/square.png'

image1 = cv2.imread(input1)  
image2 = cv2.imread(input2) 

# image inputs with applied parameters 
subs = cv2.subtract(image1, image2) 


cv2.imshow('substracted Image', subs) 

# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
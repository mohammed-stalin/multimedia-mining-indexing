import cv2  
input1='../assets/mountain.jpg'
input2='../assets/horse.jpg'

image1 = cv2.imread(input1)  
image2 = cv2.imread(input2) 

image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# image inputs with applied parameters 
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0) 


cv2.imshow('Weighted Image', weightedSum) 

# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
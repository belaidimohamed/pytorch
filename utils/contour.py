import cv2
import numpy as np

class contour():
    def __init__(self, image):
        self.image = image
    def __call__(self):
        edged = cv2.Canny(self.image, 100, 200)
        cv2.waitKey(0)

        # findContours alters the image
        _ , hierarchy = cv2.findContours(edged,	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(edged)
        cv2.imshow('Canny Edges After Contouring', edged)
        cv2.waitKey(0)
        return edged

# img = cv2.imread(r"C:\Users\mohamed\Desktop\tensor\assets\cat2.jpg" , cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img , (200,200))
# cm = Contour(img)
# cm.getContour()

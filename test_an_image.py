import cv2
import numpy as np
import network
import torch
import os

#dog [0,1] , cat [1,0]

img_resolution = 50

device = torch.device('cpu')
model = network.Net()
model.load_state_dict(torch.load(r"C:\Users\mohamed\Desktop\tensor\trained_models\0.743-dogVscat", map_location=device))

path = input("feed me the full path of the image to test :) : ")
imgg = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

if imgg == None :
    imgg = cv2.imread(os.path.join(r"C:\Users\mohamed\Desktop\tensor\assets",path),cv2.IMREAD_GRAYSCALE)

img = cv2.resize(imgg,(img_resolution,img_resolution))
img_array =  np.array(img)
img_tensor = torch.Tensor(img_array)
net_output = model(img_tensor.view(-1,1,img_resolution,img_resolution))

print(net_output[0])
print(torch.argmax(net_output[0]))
# p = torch.softmax(net_output,dim=1)
# print(p)
#
# print("dog : ", round(p[0][1].item(),3) *100 ," %")
# print("cat : ", round(p[0][0].item(),3) *100 ," %")

print("dog : ", round(np.exp(net_output[0][1].item()), 3) *100, " %")
print("cat : ", round(np.exp(net_output[0][0].item()), 3) *100, " %")

cv2.imshow("image", imgg)
cv2.waitKey(0)
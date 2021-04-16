import os
import cv2
from tqdm import tqdm
import numpy as np
import utils.contour as C
os.chdir(r'C:\Users\mohamed\Desktop\tensor')

#dog [0,1] , cat [1,0]

REBUILD_DATA = True
class CatsVsDogs:
    img_size = 75
    cats = 'PetImages/Cat'
    dogs = 'PetImages/Dog'
    labels = { dogs:1, cats:0 }
    training_data = []
    catCount = 0
    dogCount = 0
    error = 0
    def make_training_data(self):
        for label in self.labels :
                print(label)
                for f in tqdm(os.listdir(label)):
                    try :
                        path = os.path.join(label,f)
                        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img , (self.img_size,self.img_size))
                        # img = C.contour(img)
                        self.training_data.append([np.array(img, dtype="object"), np.eye(2)[self.labels[label]]])
                        if label == self.cats :
                            self.catCount +=1
                        elif label == self.dogs :
                            self.dogCount += 1
                    except Exception as e:
                        self.error+=1
                        pass
        np.random.shuffle(self.training_data)
        np.save(r'C:\Users\mohamed\Desktop\tensor\training_data_75.npy',self.training_data)
        print('cats: ',self.catCount)
        print('dogs: ',self.dogCount)
        print('error',self.error)

if REBUILD_DATA :
    cvd = CatsVsDogs()
    cvd.make_training_data()






import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image


X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)
print(nclasses)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=500,train_size=3500,random_state=0)

X_train_scale = X_train/255
X_test_scale = X_test/255

clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scale,Y_train)

def get_predict(image):
    imageEX = Image.open(image)
    imagebw = imageEX.convert('L')
    imagebw_resized = imagebw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(imagebw_resized,pixel_filter)
    imagebw_resized_inverted_scaled = np.clip(imagebw_resized-min_pixel,0,255)
    max_pixel = np.max(imagebw_resized)
    imagebw_resized_inverted_scaled = np.asarray(imagebw_resized_inverted_scaled)/max_pixel
    test_sampel = np.array(imagebw_resized_inverted_scaled).reshape(1,784)
    test_predicted = clf.predict(test_sampel)
    return test_predicted[0]
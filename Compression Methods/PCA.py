import cv2
import glob
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from google.colab.patches import cv2_imshow

!mkdir PCA_dataset #Directory of new dataset
img_dir = r"/content/flickr30k_images/" #Enter Directory of 30k_images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
disk_dir = Path("/content/PCA_dataset")
counter = 0

for f1 in files:
    if counter == 30: #First 30 images of dataset
        break;
    counter = counter + 1
    img = cv2.cvtColor(cv2.imread(f1), cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    id = os.path.split(f1)
    filename = id[1] 

    blue,green,red = cv2.split(img) 
    pca = PCA(30) #Number of components


    red_transformed = pca.fit_transform(red)
    red_inverted = pca.inverse_transform(red_transformed)
    green_transformed = pca.fit_transform(green)
    green_inverted = pca.inverse_transform(green_transformed)
    blue_transformed = pca.fit_transform(blue)
    blue_inverted = pca.inverse_transform(blue_transformed)

    img_compressed = np.dstack((red_inverted, green_inverted, blue_inverted))
    cv2.imwrite(os.path.join(disk_dir, filename), img_compressed)
    #cv2_imshow(img_compressed)

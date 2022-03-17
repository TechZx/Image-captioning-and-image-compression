import os
import glob
import numpy as np
from skimage import io
from pathlib import Path
from imageio import imsave
from sklearn.cluster import KMeans
from pylab import *
%matplotlib inline

!mkdir KM_dataset #Directory of new dataset
img_dir = r"/content/flickr30k_images/"   #Enter Directory of 30k_images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
disk_dir = Path("/content/KM_dataset")
counter = 0

for f1 in files:  
    if counter == 30: #First 30 images of dataset
        break;
    counter = counter + 1
    img = io.imread(f1)
    id = os.path.split(f1)
    filename = id[1] 

    rows, cols = img.shape[0], img.shape[1]
    img = img.reshape(rows * cols, 3)

    kMeans = KMeans(n_clusters = 16) #Number of clusters
    kMeans.fit(img)

    centers = np.asarray(kMeans.cluster_centers_, dtype = np.uint8)
    labels = np.asarray(kMeans.labels_, dtype = np.uint8)
    labels = np.reshape(labels, (rows, cols))

    compressed_image = np.zeros((rows, cols, 3), dtype = np.uint8)
    for i in range(rows):
        for j in range(cols):
              # assinging every pixel the rgb color of their label's center 
                compressed_image[i, j, :] = centers[labels[i, j], :]
    
    #io.imshow(compressed_image)
    #show()
    path_img_source = os.path.join(disk_dir, filename) 
    io.imsave( path_img_source, compressed_image)
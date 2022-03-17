import os
import cv2
import glob
import numpy as np
from pathlib import Path
from numpy.linalg import svd

!mkdir SVD_dataset #Directory of new dataset
img_dir = r"/content/flickr30k_images/"   #Enter Directory of 30k_images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
disk_dir = Path("/content/SVD_dataset")
counter = 0

for f1 in files:  
    if counter == 30: #First 30 images of dataset
        break;
    counter = counter + 1
    img = cv2.imread(f1)
    id = os.path.split(f1)
    filename = id[1]  

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    k = 30  #Number of components

    ur,sr,vr = svd(r, full_matrices=False)
    ug,sg,vg = svd(g, full_matrices=False)
    ub,sb,vb = svd(b, full_matrices=False)
    rr = np.dot(ur[:,:k],np.dot(np.diag(sr[:k]), vr[:k,:]))
    rg = np.dot(ug[:,:k],np.dot(np.diag(sg[:k]), vg[:k,:]))
    rb = np.dot(ub[:,:k],np.dot(np.diag(sb[:k]), vb[:k,:]))


    rimg = np.zeros(img.shape)
    rimg[:,:,0] = rr
    rimg[:,:,1] = rg
    rimg[:,:,2] = rb
    
    for ind1, row in enumerate(rimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    rimg[ind1,ind2,ind3] = abs(value)
                if value > 255:
                    rimg[ind1,ind2,ind3] = 255

    compressed_image = rimg.astype(np.uint8)
    cv2.imwrite(os.path.join(disk_dir, filename), compressed_image)
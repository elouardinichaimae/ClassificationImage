#%%
import glob
from patchify import  unpatchify
import numpy as np
import cv2

from matplotlib import pyplot as plt

# predicted_images= [:,:,:,:,]
# glob.glob('Data/segmentes/*.png', recursive=False)
# predicted_images=glob.glob('Data/segmentes/*.png')
# print(predicted_images)
# predicted_images=np.array(predicted_images)
# predicted_patches_reshaped = np.reshape(predicted_images, predicted_images[0],predicted_images[1],predicted_images[2],predicted_images[3])
# reconstructed_image = unpatchify(predicted_patches_reshaped, (17920,14848,3))
# plt.imshow(reconstructed_image)
# plt.axis('off')


patched_prediction= []

path = "patches/images_big/*.png"
for file in glob.glob(path):
    img1= cv2.imread(file)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    for i in range(35):
        for j in range(29):
            patched_prediction.append(img)


patched_prediction = np.array(patched_prediction)
print(patched_prediction.shape) 
patched_prediction = np.reshape(patched_prediction, [7, 8, 
                                            2048, 2048])

unpatched_prediction = unpatchify(patched_prediction, (17920, 14848))

plt.imshow(unpatched_prediction)
plt.axis('off')
cv2.imwrite('patches/final/' + 'final.png',unpatched_prediction)
#################

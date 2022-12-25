# %%
import os
from skimage import io
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# from future import print_function

#Loading a png mask image for inspection
test1_mask_png = io.imread("labels_as_png/arbre.png")
# plt.imshow(test_mask_png, cmap='gray')
# print(np.unique(test_mask_png))  #This is not a true binary image.



#Need to binarize the image. Simple thresholding for values above 0. 
#Convert all values above 0 to 1 to assign a pixel value of 1 for the arbre class.
#Similarly convert other values for other classes to 2, 3, etc. 
my_mask1 = np.where(test1_mask_png>0, 1, test1_mask_png)
print(np.unique(my_mask1))
# plt.imshow(my_mask1, cmap='gray')

# sol
test2_mask_png = io.imread("labels_as_png/sol.png")
my_mask2 = np.where(test2_mask_png>0, 2, test2_mask_png)
print(np.unique(my_mask2))
# plt.imshow(my_mask2, cmap='gray')

#tronc
test3_mask_png = io.imread("labels_as_png/tronc.png")
my_mask3 = np.where(test3_mask_png>0, 3, test3_mask_png)
print(np.unique(my_mask3))
# plt.imshow(my_mask3, cmap='gray')

#Now, let us read images from all classes and change pixel values to 1, 2, 3, ...
#You can also combine them into a single image (numpy array) for simple handling in future
#(Changing pixel values is optional if you do not intend to combine them into a single array)
#It is better to keep them separate, especially for multilabel segmentation
#where classes can overlap. 

label_folder = "labels_as_png/"
arbre_masks = []
sol_masks = []
tronc_masks = []

all_masks=[]

for filename in os.listdir(label_folder):
    #print(filename)
    if "arbre" in filename:
        print(filename)
        arbre_mask = io.imread(label_folder + filename)
        arbre_mask = np.where(arbre_mask>0, 1, arbre_mask)
        arbre_masks.append(arbre_mask)
        
for filename in os.listdir(label_folder):
    if "sol" in filename:
        print(filename)
        sol_mask = io.imread(label_folder + filename)
        sol_mask = np.where(sol_mask>0, 2, sol_mask)
        sol_masks.append(sol_mask)

for filename in os.listdir(label_folder):
    if "tronc" in filename:
        print(filename)
        tronc_mask = io.imread(label_folder + filename)
        tronc_mask = np.where(tronc_mask>0, 3, tronc_mask)
        tronc_masks.append(tronc_mask)
A = np.array(tronc_mask)
B=np.array(sol_mask)
C=np.array(arbre_mask)
print(np.unique(A))
print(np.unique(B))
print(np.unique(C))
sum= A + B + C
print(np.unique(sum))
sum[sum==4]=0
print(np.unique(sum))
plt.figure(figsize=(4.989, 8.989), dpi=100)



w = 8.989
h = 4.989

fig = plt.figure(frameon=False)
fig.set_size_inches(w,h)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(my_mask3, cmap='gray')
plt.savefig('troncslab.png', dpi=1000)

# for i in range(4989) :
#     for j in range(8989):
#             if j == 4:
#                 sum[i,j] = 0




# ar = np.array(array)
# # displaying list
# # print (array)
 


#Now, convert the list to array and proceed with your work.
#NOTE that you need to resize masks (or crop) to same size to combine them
#into numpy arrays. You need to resize both input images and masks exactly the 
#same way.
# %%

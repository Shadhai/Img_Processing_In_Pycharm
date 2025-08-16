import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib
from IPython.core.pylabtools import figsize
from matplotlib.pyplot import imshow

matplotlib.use('Agg')

img=plt.imread(r'C:\Users\Shadh\Pictures\Camera Roll\gan.jpg')
img_c=cv2.imread(r'C:\Users\Shadh\Pictures\Camera Roll\gan.jpg')
print(img.shape)
print(img_c.shape)
pd.Series(img.flatten()).plot(kind='hist',
                              bins=50,
                              title='pixel ')
plt.savefig("h.png")
#disaply
fig,ax=plt.subplots(figsize=(8,8))
ax.set_title('display')
ax.imshow(img)
plt.axis('off')
plt.savefig("dispalyed.jpeg")
#coloring
fig,axs=plt.subplots(1,3,figsize=(8,8))
axs[0].imshow(img[:,:,0],cmap='Reds')
axs[1].imshow(img[:,:,1],cmap='Greys')
axs[2].imshow(img[:,:,2],cmap='Blues')
for ax in axs:
    ax.axis('off')
# axs[0].axis('off')
# axs[1].axis('off')
# axs[2].axis('off')
plt.savefig('colured_one.png')
#cv2->BGR
# Convert BGR (cv2) to RGB for correct color display
img_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)

# Display and save
plt.imshow(img_rgb)
plt.axis('off')
plt.title('RGB Image from cv2')
plt.savefig("cv2_displayed.png")  # Save the displayed image
plt.imsave("cv2_raw.png", img_rgb)  # Save just the raw RGB image data
g2=cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
plt.imshow(g2)
plt.axis('off')
plt.title('GRAY one')
plt.imsave("GRAY.png",g2)
print(g2.shape)
#resize & scaling
img_rgb = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.resize(img_rgb, None, fx=0.25, fy=0.25)
plt.imshow(resized_rgb)
plt.axis('off')
plt.imsave("resized.png",resized_rgb)
# Blur the image
blurred = cv2.GaussianBlur(img_rgb, (7, 7), 0)
fig,axs=plt.subplots(figsize=(10,10))
plt.imshow(blurred)
plt.imsave("blur.png",blurred)
# Sharpen the image
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(img_rgb, -1, kernel)
fig,ass=plt.subplots(figsize=(10,10))
plt.imshow(sharpened)
ass.axis('off')
plt.imsave("shapren.png",sharpened)
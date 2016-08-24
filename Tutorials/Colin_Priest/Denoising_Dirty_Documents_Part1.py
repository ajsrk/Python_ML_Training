#Displaying Image Using PIL
#from PIL import Image
#im = Image.open("/home/Dhaneesh/Downloads/Denoising_Dirty_Documents/train/6.png")
#im.show()
#Plotting Image using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread("/home/Dhaneesh/Downloads/Denoising_Dirty_Documents/train/6.png")
plt.imshow(img)
plt.show()


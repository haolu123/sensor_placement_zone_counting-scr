# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:55:21 2022

@author: Admin
"""
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt

img_file = './data/fp1_code.png'
img = Image.open(img_file)

fig,ax = plt.subplots()
ax.imshow(img)

rect = patches.Rectangle((320,110), 57,57,linewidth=1, edgecolor='r', facecolor='none')

ax.add_patch(rect)
rect = patches.Rectangle((120,110), 57,57,linewidth=1, edgecolor='r', facecolor='none')

ax.add_patch(rect)
fig.show()

import os, glob,sys
import numpy as np
import cv2

import matplotlib.pyplot as plt



if __name__=="__main__":
	print("Start")
	img = cv2.imread("../data/test/test.png")
	print(img.shape)

	rows,cols,_ = img.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
	dst = cv2.warpAffine(img,M,(cols,rows))

	plt.figure()
	plt.imshow(dst[...,::-1])
	plt.show()

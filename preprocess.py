# Preprocessing the dataset
# The preprocessing should include denoising and image enhancing
# Tried blind deconvolution on Matlab, but no luck

import os
import cv2
import numpy as np

from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.segmentation import flood_fill
from skimage.restoration  import denoise_bilateral, denoise_tv_chambolle

def get_mask(img, threshold = 25, window_size = 7):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_blur = cv2.blur(img_gray, (window_size, window_size))
	img_mask = img_blur >= threshold
	img_mask = flood_fill(img_mask, (0, 0), 0)
	img_mask = flood_fill(img_mask, (-1, 0), 0)
	img_mask = flood_fill(img_mask, (0, -1), 0)
	img_mask = flood_fill(img_mask, (-1, -1), 0)
	return img_mask

def apply_mask(img, mask):
	masked_img = img.copy()
	for c in range(img.shape[2]):
		masked_img[:,:,c] = masked_img[:,:,c] * mask
	return masked_img

def sharpen(img, window_size = 3, alpha = 10):
	kernel = np.array([[0, -1, 0],
					   [-1, 5,-1],
					   [0, -1, 0]])
	#img_blur = ndimage.gaussian_filter(img, window_size)
	#img_filter_blur = ndimage.gaussian_filter(img_blur, 1)
	#sharpened = img_blur + alpha * (img_blur - img_filter_blur)
	sharpened = cv2.filter2D(src=img, ddepth = -1, kernel=kernel)
	return sharpened

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Applying pre-processing to a single sample')
	parser.add_argument('-i','--image_path', help='Path of the RGB image to preprocess', type=str)
	args = vars(parser.parse_args())

	img_path = args['image_path'] 
	img = cv2.imread(img_path)
	mask = get_mask(img)
	masked = apply_mask(img, mask)
	#masked = img * mask

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	sharpened = sharpen(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))

	fig, axs = plt.subplots(1,4)
	axs[0].imshow(img)
	axs[1].imshow(mask)
	axs[2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
	#axs[3].imshow(sharpened)
	#axs[3].imshow(denoise_bilateral(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), sigma_color=0.05, sigma_spatial=15, multichannel=True))
	denoised = denoise_tv_chambolle(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), weight=0.1, multichannel=True)
	axs[3].imshow(denoised)
	#plt.show()
	print(denoised.shape)
	#denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
	print(denoised.max())
	#sharpened = sharpen(denoised) * 255
	sharpened = sharpen(cv2.cvtColor((denoised*255).astype('uint8'), cv2.COLOR_RGB2BGR))
	print(sharpened.shape)
	cv2.imwrite('sharpened3.jpg', sharpened)

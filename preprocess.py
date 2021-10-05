# Preprocessing the dataset
# The preprocessing should include denoising and image enhancing
# Tried blind deconvolution on Matlab, but no luck

import os
import cv2
import bm3d
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


def denoise(img, method = "bm3d"):
	if method == 'bm3d':
		denoised = bm3d.bm3d(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
	elif method == 'tv':
		denoised = denoise_tv_chambolle(img, weight=0.1, multichannel=True)*255
	elif method == 'nlm':
		denoised = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),None,10,10,7,21)
		denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
	return denoised.astype('uint8')

def denoise_from_path(input_path, output_path, method):
	if input_path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
		print("Preprocessing ", input_path)
		img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
		denoised = denoise(img, method)
		#sharpened = sharpen(denoised) * 255
		denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
		cv2.imwrite(output_path, denoised)
		print("... done. Saved as: ", output_path)

def denoise_folders(input_path, output_path, method = 'bm3d'):
	folder_content = os.listdir(input_path)
	for f in folder_content:
		new_input = os.path.join(input_path, f)
		new_output = os.path.join(output_path, f)
		if os.path.isdir(new_input):
			if not os.path.exists(new_output):
				os.mkdir(new_output)
			denoise_folders(new_input, new_output)
		else:
			denoise_from_path(new_input, new_output, method)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Applying pre-processing to a single sample or directory content')
	parser.add_argument('-i','--input_path', help='Path of the RGB image/directory to preprocess', type=str)
	parser.add_argument('-o','--output_path', help='Path to save the preprocessed images', type=str)
	parser.add_argument('-m','--method', help='Denoising mode', type=str, default='bm3d')
	args = vars(parser.parse_args())

	input_path  = args['input_path'] 
	output_path = args['output_path'] 
	method      = args['method'] 

	if not os.path.isdir(input_path):
		print("Preprocessing single image")
		new_output = os.path.join(output_path, input_path.split(os.sep)[-1])
		denoise_from_path(input_path, new_output, method)
	else:
		print("Preprocessing folder")
		denoise_folders(input_path, output_path, method)

	#img = cv2.imread(img_path)
	#mask = get_mask(img)
	#masked = apply_mask(img, mask)
	#masked = img * mask

	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#sharpened = sharpen(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))

	#fig, axs = plt.subplots(1,4)
	#axs[0].imshow(img)
	#axs[1].imshow(mask)
	#axs[2].imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
	#axs[3].imshow(sharpened)
	#axs[3].imshow(denoise_bilateral(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), sigma_color=0.05, sigma_spatial=15, multichannel=True))
	#denoised = denoise_tv_chambolle(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB), weight=0.1, multichannel=True)
	#axs[3].imshow(denoised)
	#plt.show()
	#print(denoised.shape)
	#denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
	#print(denoised.max())
	#sharpened = sharpen(denoised) * 255
	#sharpened = sharpen(cv2.cvtColor((denoised*255).astype('uint8'), cv2.COLOR_RGB2BGR))
	#print(sharpened.shape)
	#cv2.imwrite('sharpened3.jpg', sharpened)

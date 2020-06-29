import PIL.Image
import skimage
import skimage.filters
import numpy as np


class Binarizer:
	def __init__(self, window_size=15):
		self._window_size = window_size

	def __call__(self, im):
		cutout = np.array(im)
		window_size = self._window_size

		if window_size is None:
			window_size = target_height // 2 - 1
		if window_size <= 0:
			try:
				thresh = skimage.filters.threshold_otsu(cutout)
			except ValueError:
				thresh = 128
		else:
			thresh = skimage.filters.threshold_sauvola(
				cutout, window_size=window_size)

		cutout = (cutout > thresh).astype(np.uint8) * 255
		return PIL.Image.fromarray(cutout)

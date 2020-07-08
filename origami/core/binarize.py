import PIL.Image
import skimage
import skimage.filters
import numpy as np

from functools import partial

from origami.core.utils import build_func_from_string


def binarize_with_threshold(im, threshold):
	pixels = np.array(im)
	t = threshold(pixels)
	pixels = (pixels > t).astype(np.uint8) * 255
	return PIL.Image.fromarray(pixels)


def otsu():
	def threshold(pixels):
		try:
			return skimage.filters.threshold_otsu(pixels)
		except ValueError:
			return 128

	return partial(
		binarize_with_threshold, threshold=threshold)


def sauvola(window_size=15):
	def threshold(pixels):
		return skimage.filters.threshold_sauvola(
			pixels, window_size=window_size)

	return partial(
		binarize_with_threshold, threshold=threshold)


def from_string(spec):
	funcs = dict(
		otsu=otsu,
		sauvola=sauvola)

	return build_func_from_string(spec, funcs)()

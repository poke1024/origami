import skimage.filters
import skimage.morphology
import numpy as np
import PIL.Image
import math
import cv2
import scipy


def make_ellipse(w, h):
	structure = skimage.morphology.disk(max(w, h)).astype(np.float32)
	structure = cv2.resize(structure, (w, h), interpolation=cv2.INTER_AREA)
	return structure / np.sum(structure)


class LineSkewEstimator:
	def __init__(self, min_length=50, kernel=(16, 8)):
		self._max_phi = 30 * (math.pi / 180)
		self._min_length = min_length
		self._kernel = kernel

	def _binarize(self, im):
		im = im.convert("L")
		pixels = np.array(im)
		thresh = skimage.filters.threshold_sauvola(pixels, window_size=15)
		binarized = (pixels > thresh).astype(np.uint8) * 255
		return binarized

	def _detect_line_regions(self, im):
		pix2 = self._binarize(im)

		# old algorithm (more parameters).
		"""
		pix2 = scipy.ndimage.morphology.binary_dilation(
			pix2, np.ones((1, 2)), iterations=2)

		pix2 = scipy.ndimage.morphology.binary_opening(
			pix2, np.ones((3, 7)), iterations=3)

		pix2 = scipy.ndimage.morphology.binary_dilation(
			pix2, np.ones((1, 2)), iterations=2)

		pix2 = scipy.ndimage.morphology.binary_opening(
			pix2, np.ones((5, 5)), iterations=1)
		"""

		pix2 = skimage.filters.sobel_h(pix2)
		pix2 = (pix2 == 1).astype(np.uint8) * 255

		pix2 = skimage.filters.sobel_h(pix2)
		pix2 = (pix2 == 1).astype(np.uint8) * 255

		pix2 = scipy.ndimage.filters.convolve(
			pix2.astype(np.float32) / 255, make_ellipse(*self._kernel))

		thresh = skimage.filters.threshold_sauvola(pix2, 3)
		pix2 = (pix2 > thresh).astype(np.uint8) * 255

		pix2 = np.logical_not(pix2)

		return pix2

	def __call__(self, im):
		pix2 = self._detect_line_regions(im)

		pix3 = skimage.measure.label(np.logical_not(pix2), background=False)
		props = skimage.measure.regionprops(pix3)

		for prop in props:
			if prop.major_axis_length < self._min_length:
				# not enough evidence
				continue
			if prop.eccentricity < 0.99:
				# not line-shaped enough
				continue

			p = prop.centroid
			phi = prop.orientation

			phi = math.acos(math.cos(phi - math.pi / 2))
			if phi > math.pi / 2:
				phi -= math.pi

			if abs(phi) > self._max_phi:
				continue

			yield p[::-1], phi

	def annotate(self, im):
		im = im.convert("RGB")
		pixels = np.array(im)
		for p, phi in self(im):
			p = tuple(map(int, p))[::-1]
			r = 50
			v = np.array([math.cos(phi), math.sin(phi)]) * r

			q = np.array(p) + v
			q = tuple(map(int, q))
			cv2.line(pixels, p, q, (255, 0, 0), 2)

			q = np.array(p) - v
			q = tuple(map(int, q))
			cv2.line(pixels, p, q, (255, 0, 0), 2)

		return PIL.Image.fromarray(pixels)

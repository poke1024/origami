import math
import numpy as np
import wquantiles
import cv2
import shapely.affinity
import PIL.Image

from origami.core.math import to_shapely_matrix


class Deskewer:
	def __init__(self, lines=None, skew=None):
		if skew is None:
			assert lines is not None
			angles = np.array([line.angle for line in lines.values()])
			lengths = np.array([line.length for line in lines.values()])
			skew = wquantiles.median(angles, lengths)
		else:
			assert skew is not None

		self._skew = skew
		self._matrix = cv2.getRotationMatrix2D(
			(0, 0), skew * (180 / math.pi), 1)
		self._shapely_matrix = to_shapely_matrix(self._matrix)

	def image(self, im):
		pixels = cv2.warpAffine(np.array(im), self._matrix, (im.width, im.height))
		return PIL.Image.fromarray(pixels)

	def shapely(self, shape):
		return shapely.affinity.affine_transform(
			shape, self._shapely_matrix)

	@property
	def skew(self):
		return self._skew

	@property
	def matrix(self):
		return self._matrix

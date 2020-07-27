# utilities for working with 2x3 matrices, as used by OpenCV.

import numpy as np
from origami.core.math import to_shapely_matrix


def p(m, x, y):
	return m.dot(np.array([x, y, 1]))


def v(m, x, y):
	return m.dot(np.array([x, y, 0]))


def mul(a, b):
	return np.vstack([a, (0, 0, 1)]).dot(np.vstack([b, (0, 0, 1)]))[:2]


def inv(a):
	# note: might also use cv2.invertAffineTransform().
	return np.linalg.inv(np.vstack([a, (0, 0, 1)]))[:2]


to_shapely = to_shapely_matrix

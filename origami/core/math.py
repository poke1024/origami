import cv2
import numpy as np
import enum


class Orientation(enum.Enum):
	def __init__(self, direction, index):
		self._direction = direction
		self._index = index

	@property
	def direction(self):
		return self._direction

	@property
	def index(self):
		return self._index

	@property
	def flipped(self):
		return Orientation._flipped[self]

	H = ((1, 0), 0)
	V = ((0, 1), 1)


Orientation._flipped = dict((
	(Orientation.H, Orientation.V),
	(Orientation.V, Orientation.H)))


def resize_transform(from_size, to_size):
	w0, h0 = from_size
	w1, h1 = to_size
	return cv2.getAffineTransform(
		np.float32([[0, 0], [w0, 0], [0, h0]]),
		np.float32([[0, 0], [w1, 0], [0, h1]]))


def to_shapely_matrix(m):
	matrix = np.zeros((12, ), dtype=m.dtype)

	matrix[0:2] = m[0, 0:2]
	matrix[3:5] = m[1, 0:2]

	matrix[8] = 1

	matrix[9] = m[0, 2]
	matrix[10] = m[1, 2]

	return matrix


def inset_bounds(bounds, fringe):
	minx, miny, maxx, maxy = bounds
	minx = min(minx + fringe, maxx)
	maxx = max(maxx - fringe, minx)
	miny = min(miny + fringe, maxy)
	maxy = max(maxy - fringe, miny)
	return minx, miny, maxx, maxy


def outset_bounds(bounds, margin):
	assert margin >= 0
	minx, miny, maxx, maxy = bounds
	return minx - margin, miny - margin, maxx + margin, maxy + margin


class Geometry:
	def __init__(self, w, h):
		self._size = (w, h)

	@property
	def size(self):
		return self._size

	@property
	def area(self):
		w, h = self.size
		return w * h

	@property
	def diameter(self):
		w, h = self.size
		return np.sqrt(w * w + h * h)

	def rel_length(self, x):
		return self.diameter * x

	def rel_area(self, a):
		return (self.diameter * a) ** 2

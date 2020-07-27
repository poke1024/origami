# utilities for reversible transformations.

from . import mat2x3
import shapely
import cv2
import numpy


def _transform_labels(labels, weights, target_size, grayscale, border):
	weights = numpy.float32(weights)
	n_labels = len(weights)

	w, h = tuple(target_size)
	counts = numpy.empty((h, w, n_labels), dtype=numpy.float32)

	masks = (labels[:, :, None] == numpy.arange(n_labels))
	for i in range(n_labels):
		counts[:, :, i] = grayscale(
			masks[:, :, i].astype(numpy.float32),
			border=1. if (i == border) else 0.)

	counts = (counts > 0.).astype(numpy.float32)

	return numpy.argmax(counts * weights, axis=-1).astype(numpy.uint8)


def _n_channels(pixels):
	if pixels.ndim == 2:
		return 1
	elif pixels.ndim == 3:
		return pixels.shape[-1]
	else:
		raise RuntimeError("could not determine number of channels in %s" % pixels.shape)


def _white(pixels):
	c = _n_channels(pixels)
	if c == 1 and pixels.dtype == numpy.uint8:
		return 255
	elif c == 3 and pixels.dtype == numpy.uint8:
		return 255, 255, 255
	elif c == 1 and pixels.dtype == numpy.float32:
		return 1
	else:
		raise RuntimeError("unsupported pixel format (%s, %s)" % (pixels.shape, pixels.dtype))


class Transform:
	def __init__(self, domain_size, target_size, matrix):
		self._domain_size = tuple(domain_size)
		self._target_size = tuple(target_size)
		self._matrix = matrix

	@property
	def inverse(self):
		return Transform(
			self._target_size,
			self._domain_size,
			mat2x3.inv(self._matrix))

	@property
	def domain_size(self):
		return self.__domain_size

	@property
	def target_size(self):
		return self._target_size

	@property
	def matrix(self):
		return self._matrix

	def mask(self, mask):
		return self.grayscale(mask.astype(numpy.float32)) > 0.0

	def grayscale(self, pixels, border=None):
		if border is None:
			border = _white(pixels)
		return cv2.warpAffine(
			pixels,
			self.matrix,
			self.target_size,
			flags=cv2.INTER_AREA,
			borderMode=cv2.BORDER_CONSTANT, borderValue=border)

	def labels(self, labels, weights=None, border=0):
		if weights is None:
			return cv2.warpAffine(
				labels,
				self.matrix,
				self.target_size,
				flags=cv2.INTER_NEAREST)
		else:
			return _transform_labels(
				labels, weights, self.target_size, self.grayscale, border)

	def geometry(self, geom):
		return shapely.affinity.affine_transform(
			geom, mat2x3.to_shapely(self.matrix))


class Rotate(Transform):
	def __init__(self, size, phi, origin=None):
		size = tuple(size)

		w, h = size
		if origin is None:
			origin = (w / 2, h / 2)
		else:
			origin = tuple(origin)

		rotate = cv2.getRotationMatrix2D(origin, phi, 1.0)

		p = [mat2x3.p(rotate, x, y) for x in (0, w) for y in (0, h)]
		p = numpy.array(p)

		minx, miny = numpy.min(p, axis=0)
		maxx, maxy = numpy.max(p, axis=0)

		target_size = (
			int(numpy.ceil(maxx - minx)),
			int(numpy.ceil(maxy - miny)))

		translate = numpy.float32([[1, 0, -minx], [0, 1, -miny]])
		matrix = mat2x3.mul(translate, rotate)

		super().__init__(size, target_size, matrix)


class Resize(Transform):
	def __init__(self, from_size, to_size):
		from_size = tuple(from_size)
		to_size = tuple(to_size)
		w0, h0 = from_size
		w1, h1 = to_size
		matrix = cv2.getAffineTransform(
			numpy.float32([[0, 0], [w0, 0], [0, h0]]),
			numpy.float32([[0, 0], [w1, 0], [0, h1]]))
		super().__init__(from_size, to_size, matrix)

	def grayscale(self, pixels, border=None):
		return cv2.resize(
			pixels,
			self._target_size,
			interpolation=cv2.INTER_AREA)


class Remap:
	def __init__(self, x, y):
		self._x = x
		self._y = y

	def grayscale(self, pixels, border=None):
		if border is None:
			border = _white(pixels)
		return cv2.remap(
			pixels,
			self._x, self._y,
			cv2.INTER_AREA,
			None,
			borderMode=cv2.BORDER_CONSTANT, borderValue=border)

		return new_im.astype(im.dtype)

	def labels(self, labels, weights, border=0):
		return _transform_labels(
			labels, weights, reversed(labels.shape), self.grayscale, border)

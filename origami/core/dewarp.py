"""
Dewarping using a vector field generated from scattered samples. Uses line skew and separator lines
for warp estimation.

The basic warping algorithm implemented here is from

D. Schneider, M. Block, and R. Rojas. 2007. Robust Document Warping with Interpolated Vector Fields.
In Proceedings of the Ninth International Conference on Document Analysis and Recognition - Volume 01
(ICDAR ’07). IEEE Computer Society, USA, 113–117.
"""

import skimage.filters
import skimage.morphology
import numpy as np
import PIL.Image
import cv2
import scipy
import math
import shapely.geometry
import shapely.ops
import matplotlib.tri

from cached_property import cached_property

from origami.batch.core.utils import read_blocks
from origami.batch.core.utils import read_separators
from origami.core.lingrid import lingrid
from origami.core.mask import Mask


def make_ellipse(w, h):
	structure = skimage.morphology.disk(max(w, h)).astype(np.float32)
	structure = cv2.resize(structure, (w, h), interpolation=cv2.INTER_AREA)
	return structure / np.sum(structure)


class LineSkewEstimator:
	def __init__(self, max_phi, min_length=50, kernel=(16, 8)):
		self._max_phi = max_phi * (math.pi / 180)
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


def subdivide(coords):
	for p, q in zip(coords, coords[1:]):
		yield p
		yield (p + q) / 2
	yield coords[-1]


class ScatteredSkewSamples:
	def __init__(self):
		self._points = []
		self._values = []

	@property
	def points(self):
		return self._points

	@property
	def values(self):
		return self._values

	@property
	def warping(self):
		if len(self._values) > 3:
			return np.std(self._values)
		else:
			return 0

	def _angles(self, sep):
		coords = np.array(list(sep.coords))

		# generate more coords since many steps further down
		# in our processing pipeline will get confused if there
		# are less than 4 or 5 points.
		coords = np.array(list(subdivide(coords)))
		coords = np.array(list(subdivide(coords)))

		v = coords[1:] - coords[:-1]
		phis = np.arctan2(v[:, 1], v[:, 0])

		inner_phis = np.convolve(phis, np.ones(2) / 2, mode="valid")
		phis = [phis[0]] + list(inner_phis) + [phis[-1]]

		return coords, phis

	def add_line_skew(self, page_path, max_phi):
		blocks = read_blocks(page_path)
		estimator = LineSkewEstimator(max_phi=max_phi)

		def add(im, pos):
			for pt, phi in estimator(im):
				self._points.append(np.array(pt) + np.array(pos))
				self._values.append(phi)

		if True:  # processing all blocks at once takes 1/2 time.
			region = shapely.ops.cascaded_union([
				block.text_area() for block in blocks.values()])

			mask = Mask(region)
			im = PIL.Image.open(page_path)
			im, pos = mask.extract_image(np.array(im), background=255)
			add(im, pos)
		else:
			for block in blocks.values():
				im, pos = block.extract_image()
				add(im, pos)

	def add_separator_skew(self, separators, sep_types):
		for path, polyline in separators.items():
			if path[1] in sep_types:
				sep_points, sep_values = self._angles(polyline)
				self._points.extend(sep_points)
				self._values.extend(sep_values)


class WarpField:
	def __init__(self, samples, size):
		warp_field = lingrid(
			samples.points, samples.values, size[0], size[1])

		warp_cos = np.cos(warp_field, dtype=np.float32)
		warp_sin = np.sin(warp_field, dtype=np.float32)
		self._warp_vec = np.dstack([warp_cos, warp_sin])

	def __call__(self, x, y):
		vec = self._warp_vec

		wx = int(x)
		wy = int(y)

		wx = min(max(wx, 0), vec.shape[0] - 1)
		wy = min(max(wy, 0), vec.shape[1] - 1)

		return vec[wx, wy]

	def estimate_extent(self, axis, limit, step_size):
		vec = self._warp_vec
		extent = 0
		for y in range(vec.shape[1 - axis]):
			p = np.array([0, y]) if axis == 0 else np.array([y, 0])
			n_steps = 1
			max_steps = (2 * limit) // step_size
			while p[axis] < limit and n_steps < max_steps:
				v = self(*p)
				p = p + v * step_size
				n_steps += 1
			extent = max(extent, n_steps)
		return extent


class Transformer:
	def __init__(self, grid, grid_res):
		h, w = grid.shape[:2]

		triangles = None
		'''
		def p(i, j):
			return i * w + j

		triangles = np.empty(((w - 1) * (h - 1) * 2, 3), dtype=np.int32)
		k = 0
		for i in range(h - 1):
			for j in range(w - 1):
				triangles[k] = (p(i + 1, j), p(i, j + 1), p(i, j))
				k += 1
				triangles[k] = (p(i + 1, j), p(i + 1, j + 1), p(i, j + 1))
				k += 1
		'''

		tri = matplotlib.tri.Triangulation(
			grid[:, :, 0].flatten(),
			grid[:, :, 1].flatten(),
			triangles)

		z = np.flip(np.dstack(np.mgrid[0:h, 0:w]), axis=-1) * grid_res

		self._ix = matplotlib.tri.LinearTriInterpolator(tri, z[:, :, 0].flatten())
		self._iy = matplotlib.tri.LinearTriInterpolator(tri, z[:, :, 1].flatten())

	def __call__(self, x, y):
		tx = self._ix(x, y).compressed()
		ty = self._iy(x, y).compressed()
		return tx, ty


class Dewarper:
	horizontal_separators = ["H"]
	vertical_separators = ["V", "T"]

	def __init__(self, page_path, grid_res=25, max_phi=10,
		add_separators=True, rescale_separators=False):

		im = PIL.Image.open(page_path)
		self._im = im

		self._width = im.width
		self._height = im.height
		self._grid_res = grid_res

		if add_separators:
			separators = read_separators(page_path)

			if rescale_separators:  # legacy mode
				sx = im.width / 1280
				sy = im.height / 2400
				separators = dict((k, shapely.affinity.scale(
					s, sx, sy, origin=(0, 0))) for k, s in separators.items())
		else:
			separators = None

		self._samples_h = ScatteredSkewSamples()
		if separators:
			self._samples_h.add_separator_skew(
				separators, Dewarper.horizontal_separators)
		self._samples_h.add_line_skew(page_path, max_phi=max_phi)

		self._samples_v = ScatteredSkewSamples()
		if separators:
			self._samples_v.add_separator_skew(
				separators, Dewarper.vertical_separators)

		self._warp_field_h = None
		self._warp_field_v = None

	@property
	def warping(self):
		return max(self._samples_h.warping, self._samples_v.warping)

	def _initialize(self):
		if self._warp_field_h is None:
			size = (self._width, self._height)

			self._warp_field_h = WarpField(
				self._samples_h, size)
			self._warp_field_v = WarpField(
				self._samples_v, size)

			self._compute_grid_h()
			self._compute_grid_hv()

	def _estimate_grid_shape(self):
		# sample the vector field here? how many more grid cells do we need
		# to get from top to bottom, or left to right?
		est_width = self._warp_field_h.estimate_extent(
			0, self._width, step_size=self._grid_res)
		est_height = self._warp_field_v.estimate_extent(
			1, self._height, step_size=self._grid_res)
		return est_height, est_width

	def _compute_grid_h(self):
		grid_shape = self._estimate_grid_shape()
		grid = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=np.float32)
		p = np.zeros((2,), dtype=np.float32)
		grid_res = self._grid_res

		warp_field_h = self._warp_field_h
		for gy in range(grid.shape[0]):
			p[:] = (0, gy * grid_res)

			for gx in range(grid.shape[1]):
				grid[gy, gx, :] = p

				p += warp_field_h(*p) * grid_res

		self._grid_h = grid

	def _compute_grid_hv(self):
		grid_h = self._grid_h
		grid_res = self._grid_res
		warp_field_v = self._warp_field_v

		rows = []
		for gy in range(grid_h.shape[0]):
			rows.append(shapely.geometry.LineString(grid_h[gy]))

		grid_hv = np.empty(grid_h.shape, dtype=np.float32)
		for gx in range(grid_h.shape[1]):
			p = grid_h[0, gx][:]

			for gy in range(grid_h.shape[0] - 1):
				grid_hv[gy, gx, :] = p

				v = warp_field_v(*p) * grid_res * 2

				ray = shapely.geometry.LineString([p, p + v])
				inter = ray.intersection(rows[gy + 1])
				if inter:
					p = np.array(list(inter.coords)[0])
				else:
					p = p + v

			grid_hv[-1, gx, :] = p

		self._grid_hv = grid_hv

	@cached_property
	def annotate(self):
		self._initialize()

		pixels = np.array(self._im)

		grid_h = self._grid_h
		grid_hv = self._grid_hv

		for gy in range(grid_h.shape[0]):
			for gx in range(grid_h.shape[1] - 1):
				ip = tuple(map(int, grid_h[gy, gx]))
				iq = tuple(map(int, grid_h[gy, gx + 1]))
				cv2.line(pixels, ip, iq, (255, 0, 0), 2)

		for gy in range(grid_hv.shape[0] - 1):
			for gx in range(grid_hv.shape[1]):
				ip = tuple(map(int, grid_hv[gy, gx]))
				iq = tuple(map(int, grid_hv[gy + 1, gx]))
				cv2.line(pixels, ip, iq, (128, 0, 0), 2)

		return PIL.Image.fromarray(pixels)

	def _full_res_grid(self, interpolation=cv2.INTER_LINEAR):
		self._initialize()
		grid = self._grid_hv
		s = self._grid_res
		h, w = grid.shape[:2]
		xs = cv2.resize(grid[:, :, 0], (w * s, h * s), interpolation=interpolation)
		ys = cv2.resize(grid[:, :, 1], (w * s, h * s), interpolation=interpolation)
		return np.dstack([xs, ys])

	@cached_property
	def dewarped(self):
		x_grid_hv = self._full_res_grid()

		return PIL.Image.fromarray(cv2.remap(
			np.array(self._im), x_grid_hv.astype(np.float32),
			None, interpolation=cv2.INTER_LINEAR))

	@cached_property
	def transformer(self):
		x_grid_hv = self._full_res_grid()
		r = self._grid_res
		return Transformer(x_grid_hv[::r, ::r], r)

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
import io
import zipfile
import json

from cached_property import cached_property
from functools import lru_cache

from origami.core.lingrid import lininterp
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


class Samples:
	horizontal_separators = ["H"]
	vertical_separators = ["V", "T"]

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
	def std(self):
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

	def add_line_skew_hq(self, blocks, lines, max_phi):
		for line in lines.values():
			if abs(line.angle) < max_phi:
				self._points.append(line.center)
				self._values.append(line.angle)

	def add_line_skew_lq(self, blocks, lines, max_phi):
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


class Field:
	def __init__(self, samples, size):
		self._size = size

		self._interp = lininterp(
			samples.points, samples.values, size[0], size[1])

	def get(self, pts):
		angles = self._interp(pts)
		dx = np.cos(angles, dtype=np.float32)
		dy = np.sin(angles, dtype=np.float32)
		return np.squeeze(np.dstack([dx, dy]), axis=0)

	def estimate_extent(self, axis, limit, step_size):
		pts = np.array([[0, y] for y in range(
			0, self._size[1 - axis], step_size)]).astype(np.float32)
		if axis != 0:
			pts = np.flip(pts, axis=-1)

		n_steps = 1
		max_steps = 2 * (1 + self._size[axis] // n_steps)

		while np.any(pts[:, axis] < limit) and n_steps < max_steps:
			pts += self.get(pts) * step_size
			n_steps += 1

		return n_steps


class Transformer:
	def __init__(self, grid, grid_res):
		h, w = grid.shape[:2]

		self._interp = scipy.interpolate.LinearNDInterpolator(
			grid.reshape((h * w, 2)),
			np.flip(np.dstack(np.mgrid[0:h, 0:w]), axis=-1).reshape((h * w, 2)) * grid_res)

	def __call__(self, x, y):
		pts = self._interp(np.squeeze(np.dstack([x, y]), axis=0))
		return pts[:, 0], pts[:, 1]


class GridFactory:
	def __init__(self, size, blocks, lines, separators, grid_res=25, max_phi=30,
		rescale_separators=False):

		self._width = size[0]
		self._height = size[1]
		self._grid_res = grid_res

		if separators is not None:
			if rescale_separators:  # legacy mode
				sx = im.width / 1280
				sy = im.height / 2400
				separators = dict((k, shapely.affinity.scale(
					s, sx, sy, origin=(0, 0))) for k, s in separators.items())
		else:
			separators = None

		self._samples_h = Samples()
		if separators:
			self._samples_h.add_separator_skew(
				separators, Samples.horizontal_separators)
		self._samples_h.add_line_skew_hq(blocks, lines, max_phi=max_phi)

		self._samples_v = Samples()
		if separators:
			self._samples_v.add_separator_skew(
				separators, Samples.vertical_separators)

	@property
	def res(self):
		return self._grid_res

	@property
	def std(self):
		return max(self._samples_h.std, self._samples_v.std)

	@cached_property
	def field_h(self):
		size = (self._width, self._height)
		return Field(self._samples_h, size)

	@cached_property
	def field_v(self):
		size = (self._width, self._height)
		return Field(self._samples_v, size)

	@cached_property
	def grid_shape(self):
		# sample the vector field here? how many more grid cells do we need
		# to get from top to bottom, or left to right?
		est_width = self.field_h.estimate_extent(
			0, self._width, step_size=self._grid_res)
		est_height = self.field_v.estimate_extent(
			1, self._height, step_size=self._grid_res)
		return est_height, est_width

	@cached_property
	def grid_h(self):
		grid_shape = self.grid_shape

		grid = np.zeros((grid_shape[0], grid_shape[1], 2), dtype=np.float32)
		grid_res = self._grid_res

		field_h = self.field_h.get

		pts = np.array([
			[0, gy * grid_res]
			for gy in range(grid.shape[0])], dtype=np.float32)
		for gx in range(grid.shape[1]):
			grid[:, gx, :] = pts
			pts += field_h(pts) * grid_res

		return grid

	@cached_property
	def grid_hv(self):
		grid_h = self.grid_h
		grid_res = self._grid_res
		field_v = self.field_v.get

		rows = []
		for gy in range(grid_h.shape[0]):
			rows.append(shapely.geometry.LineString(grid_h[gy]))

		grid_hv = np.zeros(grid_h.shape, dtype=np.float32)

		pts0 = grid_h[0]

		for gy in range(grid_h.shape[0] - 1):
			grid_hv[gy, :, :] = pts0

			pts1 = pts0 + field_v(pts0) * grid_res * 2

			for i, (p0, p1) in enumerate(zip(pts0, pts1)):
				ray = shapely.geometry.LineString([p0, p1])
				inter = ray.intersection(rows[gy + 1])
				if inter:
					pts1[i] = list(inter.coords)[0]

			pts0 = pts1

		grid_hv[-1, :, :] = pts0

		return grid_hv


class Grid:
	def __init__(self, hv, res):
		self._grid_hv = hv
		self._grid_res = res

	@cached_property
	def warping(self):
		pts = self.points("sample")
		dy = (pts[1:, :, 0] - pts[:-1, :, 1]).flatten()
		dx = (pts[:, 1:, 1] - pts[:, :-1, 0]).flatten()
		return max(np.std(dx), np.std(dy))

	@lru_cache(maxsize=8)
	def points(self, resolution="full", interpolation=cv2.INTER_LINEAR):
		if resolution == "sample":
			return self._grid_hv
		elif resolution == "full":
			grid = self._grid_hv
			s = self._grid_res
			h, w = grid.shape[:2]
			xs = cv2.resize(grid[:, :, 0], (w * s, h * s), interpolation=interpolation)
			ys = cv2.resize(grid[:, :, 1], (w * s, h * s), interpolation=interpolation)
			return np.dstack([xs, ys])
		else:
			raise ValueError(resolution)

	@property
	def resolution(self):
		return self._grid_res

	@staticmethod
	def create(*args, **kwargs):
		factory = GridFactory(*args, **kwargs)
		return Grid(factory.grid_hv, factory.res)

	@staticmethod
	def open(path):
		with zipfile.ZipFile(path, "r") as zf:
			info = json.loads(zf.read("meta.json").decode("utf8"))
			data = io.BytesIO(zf.read("data.npy"))
			grid = np.load(data, allow_pickle=False)
		grid = grid.reshape(info["shape"])
		return Grid(grid, info["cell"])

	def save(self, path, compression=zipfile.ZIP_DEFLATED):
		data = io.BytesIO()
		np.save(data, self._grid_hv.astype(np.float32), allow_pickle=False)

		info = dict(cell=self._grid_res, shape=self._grid_hv.shape)
		with zipfile.ZipFile(path, "w", compression) as zf:
			zf.writestr("data.npy", data.getvalue())
			zf.writestr("meta.json", json.dumps(info))

	@cached_property
	def transformer(self):
		x_grid_hv = self.points("full")
		r = self._grid_res
		return Transformer(x_grid_hv[::r, ::r], r)


class Dewarper:
	def __init__(self, im, grid):
		self._im = im
		self._grid = grid

	@property
	def grid(self):
		return self._grid

	@cached_property
	def annotated(self):
		pixels = np.array(self._im.convert("RGB"))

		grid_hv = self._grid.points("sample")

		for gy in range(grid_hv.shape[0]):
			for gx in range(grid_hv.shape[1] - 1):
				ip = tuple(map(int, grid_hv[gy, gx]))
				iq = tuple(map(int, grid_hv[gy, gx + 1]))
				cv2.line(pixels, ip, iq, (255, 0, 0), 2)

		for gy in range(grid_hv.shape[0] - 1):
			for gx in range(grid_hv.shape[1]):
				ip = tuple(map(int, grid_hv[gy, gx]))
				iq = tuple(map(int, grid_hv[gy + 1, gx]))
				cv2.line(pixels, ip, iq, (128, 0, 0), 2)

		return PIL.Image.fromarray(pixels)

	@cached_property
	def dewarped(self):
		x_grid_hv = self._grid.points("full")

		return PIL.Image.fromarray(cv2.remap(
			np.array(self._im), x_grid_hv.astype(np.float32),
			None, interpolation=cv2.INTER_LINEAR))

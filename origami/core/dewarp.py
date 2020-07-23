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
import scipy.interpolate
import math
import shapely.geometry
import shapely.ops
import shapely.strtree
import io
import zipfile
import json
import logging
import multiprocessing

from cached_property import cached_property
from functools import lru_cache
from sklearn.decomposition import PCA

from origami.core.lingrid import lininterp
from origami.core.mask import Mask
from origami.core.math import Geometry, divide_path


class LineDetector:
	def __init__(self):
		pass

	def binarize(self, im, window=15):
		im = im.convert("L")
		pixels = np.array(im)
		thresh = skimage.filters.threshold_sauvola(pixels, window_size=window)
		binarized = (pixels > thresh).astype(np.uint8) * 255
		return binarized


class OpeningLineDetector(LineDetector):
	def __call__(self, im):
		pix2 = self.binarize(im)

		pix2 = scipy.ndimage.morphology.binary_dilation(
			pix2, np.ones((1, 2)), iterations=2)

		pix2 = scipy.ndimage.morphology.binary_opening(
			pix2, np.ones((3, 7)), iterations=3)

		pix2 = scipy.ndimage.morphology.binary_dilation(
			pix2, np.ones((1, 2)), iterations=2)

		pix2 = scipy.ndimage.morphology.binary_opening(
			pix2, np.ones((5, 5)), iterations=1)

		return pix2


class SobelLineDetector(LineDetector):
	def __init__(self, kernel=(16, 8)):
		self._kernel_size = kernel
		self._ellipse = self._make_ellipse()

	def _make_ellipse(self):
		w, h = self._kernel_size
		structure = skimage.morphology.disk(max(w, h)).astype(np.float32)
		structure = cv2.resize(structure, (w, h), interpolation=cv2.INTER_AREA)
		return structure / np.sum(structure)

	def __call__(self, im):
		pix2 = self.binarize(im)

		pix2 = skimage.filters.sobel_h(pix2)
		pix2 = (pix2 == 1).astype(np.uint8) * 255

		pix2 = skimage.filters.sobel_h(pix2)
		pix2 = (pix2 == 1).astype(np.uint8) * 255

		pix2 = scipy.ndimage.filters.convolve(
			pix2.astype(np.float32) / 255, self._ellipse)

		thresh = skimage.filters.threshold_sauvola(pix2, 3)
		pix2 = (pix2 > thresh).astype(np.uint8) * 255

		pix2 = np.logical_not(pix2)

		return pix2


class OcropyLineDetector(LineDetector):
	def __init__(self, maxcolseps=3):
		self._maxcolseps = maxcolseps

	def __call__(self, im):
		from ocrd_cis.ocropy.common import compute_hlines, compute_segmentation, binarize

		im_bin, _ = binarize(np.array(im).astype(np.float32) / 255)
		segm = compute_segmentation(im_bin, fullpage=True, maxcolseps=self._maxcolseps)
		llabels, hlines, vlines, images, colseps, scale = segm
		return hlines


class LineSkewEstimator:
	def __init__(self, line_det, max_phi, min_length=50, eccentricity=0.99):
		self._line_detector = line_det
		self._max_phi = max_phi * (math.pi / 180)
		self._min_length = min_length
		self._eccentricity = eccentricity

	def __call__(self, im):
		pix2 = self._line_detector(im)

		pix3 = skimage.measure.label(np.logical_not(pix2), background=False)
		props = skimage.measure.regionprops(pix3)

		for prop in props:
			if prop.major_axis_length < self._min_length:
				# not enough evidence
				continue
			if prop.eccentricity < self._eccentricity:
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


class BorderEstimator:
	def __init__(self, page, blocks, separators):
		self._page = page

		regions = [b.image_space_polygon for b in blocks.values()]
		separators = list(separators.values()) if separators else []
		hull = shapely.ops.cascaded_union(
			regions + separators).convex_hull

		coords = np.array(list(hull.exterior.coords))

		while np.all(coords[0] == coords[-1]):
			coords = coords[:-1]

		dx = np.abs(np.diff(coords[:, 0], append=coords[0, 0]))
		dy = np.abs(np.diff(coords[:, 1], append=coords[0, 1]))

		self._coords = coords
		self._vertical = dy - dx > 0

	@cached_property
	def unfiltered_paths(self):
		coords = self._coords
		mask = self._vertical

		if np.min(mask) == np.max(mask):
			return []

		r = 0
		while not mask[r]:
			r += 1

		rmask = np.roll(mask, -r)
		rcoords = np.roll(coords, -r, axis=0)

		paths = []
		cur = None

		for i in range(rmask.shape[0]):
			if rmask[i]:
				if cur is None:
					cur = []
					paths.append(cur)
				cur.append(rcoords[i])
			else:
				cur = None

		return paths

	def filtered_paths(self, margin=0.01, max_variance=1e-5):
		paths = self.unfiltered_paths

		pca = PCA(n_components=2)

		w, h = self._page.warped.size

		def good(path):
			pca.fit(path * (1 / w, 1 / h))
			if min(pca.explained_variance_) > max_variance:
				return False

			if np.max(path[:, 0]) / w > 1 - margin:
				return False
			elif np.min(path[:, 0]) / w < margin:
				return False
			return True

		return list(filter(good, map(np.array, paths)))

	def paths(self, **kwargs):
		paths = self.filtered_paths(**kwargs)

		def downward(path):
			if path[-1, 1] < path[0, 1]:
				return path[::-1]
			else:
				return path

		return list(map(downward, paths))


def subdivide(coords):
	for p, q in zip(coords, coords[1:]):
		yield p
		yield (p + q) / 2
	yield coords[-1]


class Samples:
	horizontal_separators = ["H"]
	vertical_separators = ["V", "T"]

	def __init__(self, geometry):
		self._points = []
		self._values = []
		self._geometry = geometry

	def __len__(self):
		return len(self._points)

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

	def print_stats(self):
		print(np.min(self._values), np.max(self._values))

	def _angles(self, coords, max_segment=0.05):
		coords = np.array(coords)

		# normalize. need to check against direction here.
		# if coords[0, 1] > coords[-1, 1]:
		#	coords = coords[::-1]

		coords = divide_path(
			coords, self._geometry.rel_length(max_segment))

		# generate more coords since many steps further down
		# in our processing pipeline will get confused if there
		# are less than 4 or 5 points.

		while len(coords) < 6:
			coords = np.array(list(subdivide(coords)))

		v = coords[1:] - coords[:-1]
		phis = np.arctan2(v[:, 1], v[:, 0])

		inner_phis = np.convolve(phis, np.ones(2) / 2, mode="valid")
		phis = [phis[0]] + list(inner_phis) + [phis[-1]]

		return coords, phis

	def add_line_skew_hq(self, blocks, lines, max_phi, delta=0):
		n_skipped = 0
		for line in lines.values():
			if abs(line.angle) < max_phi:
				self._points.append(line.center)
				self._values.append(line.angle + delta)
			else:
				n_skipped += 1
		if n_skipped > 0:
			logging.warning("skipped %d lines." % n_skipped)

	def add_line_skew_lq(self, blocks, lines, max_phi):
		estimator = LineSkewEstimator(
			line_det=SobelLineDetector(),
			max_phi=max_phi)

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

	def add_separator_skew(self, separators, sep_types, max_std=0.1):
		for path, polyline in separators.items():
			if path[1] in sep_types:
				sep_points, sep_values = self._angles(polyline.coords)

				# in rare cases, separator detection goes wrong and will
				# produce separators with mixed up coordinate order, e.g.
				#  ----><-----. we sort out these cases by assuming some
				#  maximum variance for the good ones.
				std = np.std(sep_values)
				if std > max_std:
					logging.warning(
						"ignored suspicious separator %s with std=%.1f" % (
							str(path), std))
					continue

				self._points.extend(sep_points)
				self._values.extend(sep_values)

	def add_border_skew(self, page, blocks, separators, **kwargs):
		estimator = BorderEstimator(page, blocks, separators)
		for coords in estimator.paths(**kwargs):
			sep_points, sep_values = self._angles(coords)
			self._points.extend(sep_points)
			self._values.extend(sep_values)


class Field:
	def __init__(self, samples, size):
		self._size = size

		self._interp = lininterp(
			samples.points, samples.values, (0, 0, size[0], size[1]))

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

		if n_steps >= max_steps:
			raise RuntimeError("n_steps exceeded %d" % max_steps)

		return n_steps


class Transformer:
	def __init__(self, grid, grid_res):
		h, w = grid.shape[:2]

		source = grid.reshape((h * w, 2))
		target = np.flip(np.dstack(
			np.mgrid[0:h, 0:w]), axis=-1).reshape((h * w, 2)) * grid_res

		minx = np.min(source[:, 0])
		miny = np.min(source[:, 1])
		maxx = np.max(source[:, 0])
		maxy = np.max(source[:, 1])

		self._interp = lininterp(source, target, (minx, miny, maxx, maxy))

	def __call__(self, x, y):
		pts = self._interp(np.squeeze(np.dstack([x, y]), axis=0))
		assert not np.any(np.isnan(pts))
		return pts[:, 0], pts[:, 1]


def extrapolate(a, b, x):
	v = b - a
	v /= np.linalg.norm(v)
	return b + x * v


def make_slices(n, k):
	s = math.ceil(n / k)
	for i in range(0, n, s):
		yield slice(i, i + s)


class ShapelyBatchIntersections:
	def __init__(self, grid_h, grid_res):
		self.grid_h = grid_h
		self.grid_res = grid_res

		self._rows = [
			self.make_row(gy + 1)
			for gy in range(grid_h.shape[0] - 1)]

	def make_row(self, gy, k=3):
		grid_h = self.grid_h
		grid_w = grid_h.shape[1]
		large = grid_w * self.grid_res

		def ls_for_i(i):
			pts = grid_h[gy, min(i, grid_w - k):i + k]
			if i == 0:
				pts[0] = extrapolate(pts[1], pts[0], large)
			elif i + k >= grid_w:
				pts[-1] = extrapolate(pts[-2], pts[-1], large)
			return pts

		lines = shapely.geometry.MultiLineString([
			ls_for_i(i)
			for i in range(0, grid_w, k - 1)])
		return shapely.strtree.STRtree(lines)

	def __call__(self, pts0, pts1, gy):
		row = self._rows[gy]
		ls = shapely.geometry.LineString
		norm = np.linalg.norm

		for i, (p0, p1) in enumerate(zip(pts0, pts1)):
			ray = ls([p0, p1])
			inter_pts = []
			for candidate in row.query(ray):
				inter = ray.intersection(candidate)
				if inter and not inter.is_empty:
					geom_type = inter.geom_type
					if geom_type == "Point":
						inter_pts.append(np.asarray(inter))
					elif geom_type == "MultiPoint":
						inter_pts.extend(np.asarray(inter))
					else:
						raise RuntimeError(
							"unexpected geom_type %s" % geom_type)

			if not inter_pts:
				logging.warning(
					"failed to find intersection for i=%d, n=%d." % (i, len(pts0)))
			elif len(inter_pts) == 1:
				pts1[i] = inter_pts[0]
			else:
				dist = norm(np.array(inter_pts) - p0, axis=1)
				pts1[i] = inter_pts[np.argmin(dist)]


class BentleyOttmanBatchIntersections:
	def __init__(self, grid_h, grid_res):
		from bentley_ottmann.planar import segments_intersections

		self.grid_h = grid_h
		self.intersections = segments_intersections

	def __call__(self, pts0, pts1, gy):
		grid_h = self.grid_h
		n = len(pts0)

		segments = list(zip(pts0, pts1)) + list(zip(grid_h[gy, 0:], grid_h[gy, 1:]))
		segments = [((x1, y1), (x2, y2)) for ((x1, y1), (x2, y2)) in segments]

		for pt, pairs in self.intersections(
			segments, accurate=False, validate=False).items():
			for a, b in pairs:
				if a < n <= b:
					pts1[a] = pt
				elif b < n <= a:
					pts1[b] = pt


BatchIntersections = ShapelyBatchIntersections


class GridFactory:
	def __init__(
			self, page, blocks, lines, separators,
			grid_res=25, max_phi=30, max_std=0.1,
			rescale_separators=False,
			max_grid_size=1000,
			num_threads=2):

		size = page.warped.size

		self._width = size[0]
		self._height = size[1]
		self._grid_res = grid_res
		self._max_grid_size = max_grid_size
		self._num_threads = num_threads

		if separators is not None and rescale_separators:  # legacy mode
			sx = self._width / 1280
			sy = self._height / 2400
			separators = dict((k, shapely.affinity.scale(
				s, sx, sy, origin=(0, 0))) for k, s in separators.items())

		self._samples_h = Samples(page.geometry(False))
		self._samples_v = Samples(page.geometry(False))

		if separators:
			self._samples_h.add_separator_skew(
				separators, Samples.horizontal_separators, max_std=max_std)
			self._samples_v.add_separator_skew(
				separators, Samples.vertical_separators, max_std=max_std)

		if lines:
			self._samples_h.add_line_skew_hq(
				blocks, lines, max_phi=max_phi, delta=0)
			self._samples_v.add_line_skew_hq(
				blocks, lines, max_phi=max_phi, delta=math.pi / 2)

		#self._samples_v.add_border_skew(page, blocks, separators)

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
		if max(est_width, est_height) > self._max_grid_size:
			raise RuntimeError(
				"estimated grid too big: (%d, %d)" % (est_height, est_width))
		return est_height, est_width

	def extend_border_h(self, grid_hv, side):
		field_h = self.field_h.get
		grid_res = self._grid_res
		max_borders = self._max_grid_size // 2

		if side == "left":
			def cond(x):
				return np.any(x[:, 0, 0] > 0)

			def concat(x):
				return list(reversed(x)) + [grid_hv]

			step = grid_res * -1
			borders = [grid_hv[:, :1]]
		elif side == "right":
			def cond(x):
				return np.any(x[:, -1, 0] < self._width)

			def concat(x):
				return [grid_hv] + x

			step = grid_res * 1
			borders = [grid_hv[:, -1:]]
		else:
			raise ValueError(side)

		while cond(borders[-1]):
			if len(borders) >= max_borders:
				raise RuntimeError("border extension not terminating")

			pts = borders[-1][:, 0]
			new_pts = pts + field_h(pts) * step

			borders.append(new_pts.reshape((grid_hv.shape[0], 1, 2)))

		if len(borders) > 1:
			return np.hstack(concat(borders[1:]))
		else:
			return grid_hv

	def extend_border_v(self, grid_hv, side):
		field_v = self.field_v.get
		grid_res = self._grid_res
		max_borders = self._max_grid_size // 2

		if side == "top":
			def cond(x):
				return np.any(x[0, :, 1] > 0)

			def concat(x):
				return list(reversed(x)) + [grid_hv]

			step = grid_res * -1
			borders = [grid_hv[:1, :]]
		elif side == "bottom":
			def cond(x):
				return np.any(x[-1, :, 1] < self._height)

			def concat(x):
				return [grid_hv] + x

			step = grid_res * 1
			borders = [grid_hv[-1:, :]]
		else:
			raise ValueError(side)

		while cond(borders[-1]):
			if len(borders) >= max_borders:
				raise RuntimeError("border extension not terminating")

			pts = borders[-1][0, :]
			new_pts = pts + field_v(pts) * step

			borders.append(new_pts.reshape((1, grid_hv.shape[1], 2)))

		if len(borders) > 1:
			return np.vstack(concat(borders[1:]))
		else:
			return grid_hv

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

		assert not np.any(np.isnan(grid))

		return grid

	@cached_property
	def grid_hv(self):
		grid_h = self.grid_h
		grid_res = self._grid_res
		field_v = self.field_v.get

		grid_hv = np.zeros(grid_h.shape, dtype=np.float32)
		intersections = BatchIntersections(grid_h, grid_res)

		def compute_slice(sel):
			pts0 = grid_h[0][sel]
			for gy in range(grid_h.shape[0] - 1):
				grid_hv[gy, sel, :] = pts0

				pts1 = pts0 + field_v(pts0) * grid_res * 2
				intersections(pts0, pts1, gy)
				pts0 = pts1

			grid_hv[-1, sel, :] = pts0

		if self._num_threads < 2:
			compute_slice(slice(0, grid_h.shape[1]))
		else:
			slices = make_slices(
				n=grid_h.shape[1],
				k=self._num_threads)

			with multiprocessing.pool.ThreadPool(
				processes=self._num_threads) as pool:
				pool.map(compute_slice, slices)

		grid_hv = self.extend_border_h(grid_hv, "left")
		grid_hv = self.extend_border_h(grid_hv, "right")

		grid_hv = self.extend_border_v(grid_hv, "top")
		grid_hv = self.extend_border_v(grid_hv, "bottom")

		assert not np.any(np.isnan(grid_hv))

		return grid_hv


class Grid:
	def __init__(self, hv, res):
		self._grid_hv = hv
		self._grid_res = res

	@property
	def geometry(self):
		h, w = self._grid_hv.shape[:2]
		r = self._grid_res
		return Geometry(w * r, h * r)

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

		info = dict(version=1, cell=self._grid_res, shape=self._grid_hv.shape)
		with zipfile.ZipFile(path, "w", compression) as zf:
			zf.writestr("data.npy", data.getvalue())
			zf.writestr("meta.json", json.dumps(info))

	@cached_property
	def transformer(self):
		x_grid_hv = self.points("full")
		r = self._grid_res
		return Transformer(x_grid_hv[::r, ::r], r)

	@cached_property
	def inverse_yx(self):
		grid = self.points("full")

		return scipy.interpolate.RegularGridInterpolator(
			(np.arange(grid.shape[0]), np.arange(grid.shape[1])),
			grid, method="linear", bounds_error=False, fill_value=None)

	@cached_property
	def inverse(self):
		interp = self.inverse_yx

		def f(pts):
			return interp(np.flip(pts, axis=-1))

		return f

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

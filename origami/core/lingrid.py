"""
Robust linear interpolation and extrapolation
on a grid using scattered and few data points.
"""

import numpy as np
import scipy.interpolate
import scipy.spatial
import shapely.geometry
import sympy

from enum import Enum
from cached_property import cached_property


def lerp(a, b, x):
	return (1 - x) * a + x * b


class Border(Enum):
	LEFT = 1
	TOP = 2
	BOTTOM = 3
	RIGHT = 4


class Box:
	def __init__(self, minx, miny, maxx, maxy, margin=1):
		box_pts = [
			(minx - margin, miny - margin),
			(maxx + margin, miny - margin),
			(maxx + margin, maxy + margin),
			(minx - margin, maxy + margin)]

		self._minx = box_pts[0][0]
		self._miny = box_pts[0][1]
		self._maxx = box_pts[2][0]
		self._maxy = box_pts[2][1]

		self._box = shapely.geometry.LinearRing(box_pts)
		self._box_size = max(maxx - minx, maxy - miny) + 2 * margin + 1

		self._points = []
		self._borders = dict((b, []) for b in Border)
		self._corners = dict()

	def _add(self, borders, x, y, val):
		pt = np.hstack([[x, y], val]).astype(np.float64)

		self._points.append(pt)

		for b in borders:
			self._borders[b].append(pt)

	@property
	def points(self):
		for pt in self._points:
			yield pt
		for pt, val in self._corners.items():
			yield np.hstack([pt, val])

	def _intersect(self, p, dir):
		ray = shapely.geometry.LineString([
			p, p + dir * self._box_size])
		int_pt = ray.intersection(self._box)
		if int_pt is None or int_pt.is_empty:
			raise ValueError(
				"point %s outside given domain, %s does not hit %s" % (
					p, ray, self._box))

		pt = tuple(list(int_pt.coords)[0])
		borders = set()

		is_minx = abs(pt[0] - self._minx) == 0
		is_maxx = abs(pt[0] - self._maxx) == 0
		is_miny = abs(pt[1] - self._miny) == 0
		is_maxy = abs(pt[1] - self._maxy) == 0

		if is_minx:
			borders.add(Border.LEFT)
		elif is_maxx:
			borders.add(Border.RIGHT)
		if is_miny:
			borders.add(Border.TOP)
		elif is_maxy:
			borders.add(Border.BOTTOM)

		return borders, pt[0], pt[1]

	def add_projection(self, a, b):
		a = a.astype(np.float64)
		b = b.astype(np.float64)
		v = a[:2] - b[:2]
		if np.linalg.norm(v) < 1e-2:
			return  # ignore
		normal = np.array([-v[1], v[0]])
		normal /= np.linalg.norm(normal)
		self._add(*self._intersect(a[:2], normal), a[2:])
		self._add(*self._intersect(b[:2], normal), b[2:])

	def _add_corner(self, cx, cy, p1, p2):
		if p1 is None and p2 is None:
			return

		if p1 is None:
			val = p2[2:]
		elif p2 is None:
			val = p1[2:]
		else:
			line = sympy.geometry.Line(
				sympy.geometry.Point(*p1[:2]),
				sympy.geometry.Point(*p2[:2]))
			p = line.projection(sympy.geometry.Point(cx, cy))
			p = p.evalf()
			p = np.array([p.x, p.y], dtype=np.float64)

			d = np.linalg.norm(p - p1[:2])
			d_total = np.linalg.norm(p2[:2] - p1[:2])

			if d_total == 0:
				return

			val = lerp(p1[2:], p2[2:], d / d_total)

		self._corners[(cx, cy)] = val

	def _nearest_to_corner(self, border, f, axis):
		if not self._borders[border]:
			return None
		d = dict(x=0, y=1)[axis]
		return f(self._borders[border], key=lambda pt: pt[d])

	def add_corners(self):
		# top left
		self._add_corner(
			self._minx,
			self._miny,
			self._nearest_to_corner(Border.LEFT, min, "y"),
			self._nearest_to_corner(Border.TOP, min, "x"))

		# top right
		self._add_corner(
			self._maxx,
			self._miny,
			self._nearest_to_corner(Border.RIGHT, min, "y"),
			self._nearest_to_corner(Border.TOP, max, "x"))

		# bottom right
		self._add_corner(
			self._maxx,
			self._maxy,
			self._nearest_to_corner(Border.RIGHT, max, "y"),
			self._nearest_to_corner(Border.BOTTOM, max, "x"))

		# bottom left
		self._add_corner(
			self._minx,
			self._maxy,
			self._nearest_to_corner(Border.LEFT, max, "y"),
			self._nearest_to_corner(Border.BOTTOM, min, "x"))


class Interpolator:
	def __init__(self, inter, extra, bounds):
		self._inter = inter
		self._extra = extra
		self._bounds = bounds

	def __call__(self, pts):
		pts = np.array(pts)
		if len(pts.shape) == 1:
			pts = pts[np.newaxis, :]

		minx, miny, maxx, maxy = self._bounds

		pts[:, 0] = np.clip(pts[:, 0], minx, maxx)
		pts[:, 1] = np.clip(pts[:, 1], miny, maxy)

		if self._inter is None:
			return self._extra(pts)
		else:
			ri = self._inter(pts)
			rx = self._extra(pts)
			return np.where(np.isnan(ri), rx, ri)


class InterpolatorFactory:
	def __init__(self, points, values, bounds):
		self._points = points
		self._values = values
		self._bounds = bounds

		box = Box(*self._bounds)

		if not isinstance(values[0], np.ndarray):
			self._squeeze = True
		else:
			self._squeeze = False

		try:
			hull = scipy.spatial.ConvexHull(points)
			hull_pts = list(hull.points[hull.vertices])
			self._is_collinear = False
		except scipy.spatial.qhull.QhullError:
			# most probably points are collinear.
			self._is_collinear = True

		if not self._is_collinear:
			values_dict = dict(zip([tuple(p) for p in points], values))
			hull_val = [values_dict[tuple(p)] for p in hull_pts]
			hull_pts_val = list(zip(hull_pts, hull_val))

			extra_pts = hull_pts[:]
			extra_val = hull_val[:]

			for (a, va), (b, vb) in zip(hull_pts_val, hull_pts_val[1:] + [hull_pts_val[0]]):
				box.add_projection(
					np.hstack([a, va]),
					np.hstack([b, vb]))
		else:
			pts_val = list(zip(points, values))
			for (pa, va), (pb, vb) in zip(pts_val, pts_val[1:]):
				a = np.hstack([pa, va])
				b = np.hstack([pb, vb])
				box.add_projection(a, b)
				box.add_projection(b, a)

			extra_pts = []
			extra_val = []

		box.add_corners()

		for pt in box.points:
			extra_pts.append(pt[:2])
			extra_val.append(pt[2:])

		self._extra_pts = np.array(extra_pts)
		self._extra_val = np.array(extra_val)

	@cached_property
	def grid(self):
		minx, miny, maxx, maxy = self._bounds
		assert minx == 0 and miny == 0

		grid = np.dstack(np.mgrid[0:maxx + 1, 0:maxy + 1])

		extra_pixels = scipy.interpolate.griddata(
			self._extra_pts, self._extra_val, grid,
			method="linear",
			fill_value=np.nan)

		if not self._is_collinear:
			inter_pixels = scipy.interpolate.griddata(
				self._points, self._values, grid,
				method="linear", fill_value=np.nan)

			if len(extra_pixels.shape) > 2:
				mask = np.isnan(inter_pixels[:, :, 0])
				pixels = np.empty(extra_pixels.shape)
				for i in range(extra_pixels.shape[-1]):
					pixels[:, :, i] = np.where(
						mask, extra_pixels[:, :, i], inter_pixels[:, :, i])
			else:
				mask = np.isnan(inter_pixels)
				pixels = np.where(mask, extra_pixels, inter_pixels)

		if self._squeeze and len(pixels.shape) > 2:
			pixels = pixels.squeeze(axis=-1)

		return pixels

	@cached_property
	def interpolator(self):
		extra = scipy.interpolate.LinearNDInterpolator(
			self._extra_pts, self._extra_val, fill_value=np.nan)
		if not self._is_collinear:
			inter = scipy.interpolate.LinearNDInterpolator(
				self._points, self._values, fill_value=np.nan)
		else:
			inter = None
		return Interpolator(inter, extra, self._bounds)


def lingrid(points, values, width, height):
	return InterpolatorFactory(points, values, (0, 0, width - 1, height - 1)).grid


def lininterp(points, values, bounds):
	return InterpolatorFactory(points, values, bounds).interpolator

"""
Robust linear interpolation and extrapolation
on a grid using scattered and few data points.
"""

import numpy as np
import scipy.interpolate
import scipy.spatial
import sympy

from enum import Enum


def lerp(a, b, x):
	return (1 - x) * a + x * b


class Border(Enum):
	LEFT = 1
	TOP = 2
	BOTTOM = 3
	RIGHT = 4


class Box:
	def __init__(self, width, height):
		margin = 1

		box_pts = [sympy.geometry.Point(x, y) for x, y in [
			(-margin, -margin),
			(width + margin, -margin),
			(width + margin, height + margin),
			(-margin, height + margin)]]

		self._minx = box_pts[0].x
		self._miny = box_pts[0].y
		self._maxx = box_pts[2].x
		self._maxy = box_pts[2].y

		self._box = sympy.geometry.polygon.Polygon(*box_pts)

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

	def _intersect(self, a, b):
		ray = sympy.geometry.Ray(
			sympy.geometry.Point(*a),
			sympy.geometry.Point(*b))
		intersection = sympy.geometry.intersection(
			ray, self._box)
		assert len(intersection) >= 1

		pt = intersection[0]
		borders = set()
		if pt.x == self._minx:
			borders.add(Border.LEFT)
		elif pt.x == self._maxx:
			borders.add(Border.RIGHT)
		if pt.y == self._miny:
			borders.add(Border.TOP)
		elif pt.y == self._maxy:
			borders.add(Border.BOTTOM)

		fpt = pt.evalf()
		return borders, fpt.x, fpt.y

	def add_projection(self, a, b):
		a = a.astype(np.float64)
		b = b.astype(np.float64)
		v = a[:2] - b[:2]
		if np.linalg.norm(v) < 1e-2:
			return  # ignore
		normal = np.array([-v[1], v[0]])
		normal /= np.linalg.norm(normal)
		self._add(*self._intersect(a[:2], a[:2] + normal), a[2:])
		self._add(*self._intersect(b[:2], b[:2] + normal), b[2:])

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
			self._minx.evalf(),
			self._miny.evalf(),
			self._nearest_to_corner(Border.LEFT, min, "y"),
			self._nearest_to_corner(Border.TOP, min, "x"))

		# top right
		self._add_corner(
			self._maxx.evalf(),
			self._miny.evalf(),
			self._nearest_to_corner(Border.RIGHT, min, "y"),
			self._nearest_to_corner(Border.TOP, max, "x"))

		# bottom right
		self._add_corner(
			self._maxx.evalf(),
			self._maxy.evalf(),
			self._nearest_to_corner(Border.RIGHT, max, "y"),
			self._nearest_to_corner(Border.BOTTOM, max, "x"))

		# bottom left
		self._add_corner(
			self._minx.evalf(),
			self._maxy.evalf(),
			self._nearest_to_corner(Border.LEFT, max, "y"),
			self._nearest_to_corner(Border.BOTTOM, min, "x"))


def lingrid(points, values, width, height):
	box = Box(width, height)

	if not isinstance(values[0], np.ndarray):
		squeeze = True
	else:
		squeeze = False

	try:
		hull = scipy.spatial.ConvexHull(points)
		hull_pts = list(hull.points[hull.vertices])
		is_collinear = False
	except scipy.spatial.qhull.QhullError:
		# most probably points are collinear.
		is_collinear = True

	if not is_collinear:
		values_dict = dict(zip([tuple(p) for p in points], values))
		hull_val = [values_dict[tuple(p)] for p in hull_pts]
		hull_pts_val = list(zip(hull_pts, hull_val))

		extra_pts = hull_pts[:]
		extra_val = hull_val[:]

		for (a, va), (b, vb) in zip(hull_pts_val, hull_pts_val[1:] + [hull_pts_val[0]]):
			box.add_projection(
				np.hstack([a, va]),
				np.hstack([b, vb])
			)
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

	extra_pts = np.array(extra_pts)
	extra_val = np.array(extra_val)

	grid = np.dstack(np.mgrid[0:width, 0:height])

	extra_pixels = scipy.interpolate.griddata(
		extra_pts, extra_val, grid,
		method="linear",
		fill_value=np.nan)

	if not is_collinear:
		inter_pixels = scipy.interpolate.griddata(
			points, values, grid,
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

	if squeeze and len(pixels.shape) > 2:
		pixels = pixels.squeeze(axis=-1)

	return pixels

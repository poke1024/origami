import shapely.ops
import cv2
import rtree
import logging
import numpy as np

from origami.core.contours import find_contours
from origami.train.segment.gen.masks import mask_to_polygons, polygons_to_mask


def contours(mask, convex=False):
	contours = find_contours(mask.astype(np.uint8))

	for c in contours:
		if len(c) < 3:
			continue

		if convex:
			hull = cv2.convexHull(c, returnPoints=False)
			hull = hull.reshape((hull.shape[0],))
			pts = c.reshape(c.shape[0], 2)[hull]
		else:
			pts = c.reshape(c.shape[0], 2)

		polygon = shapely.geometry.Polygon(pts)

		if not polygon.is_empty:
			yield polygon


def convex_contours(mask):
	return contours(mask, convex=True)


class Simplifier:
	def __init__(self, simplify=3, eps_area=100):
		self._simplify = simplify
		self._eps_area = eps_area

	def __call__(self, polygons):
		for polygon in polygons:
			if self._simplify is not None:
				polygon = polygon.simplify(
					self._simplify, preserve_topology=False)

				if polygon.is_empty:
					continue

			if self._eps_area is not None:
				minx, miny, maxx, maxy = polygon.bounds
				if (maxx - minx) * (maxy - miny) < self._eps_area:
					continue

				area = polygon.area
				if area < self._eps_area:
					continue

			yield polygon


def convex_hull(polygons):
	return shapely.ops.cascaded_union(list(polygons)).convex_hull


def regions_to_convex_hull(mask):
	polygons = mask_to_polygons(mask, convex_hulls=True)
	polygons = [polygon for polygon in polygons if polygon.area > 100]
	return polygons_to_mask(mask.shape, polygons)


def merge_convex_all(polygons):
	polygons = [polygon for polygon in polygons if not polygon.is_empty]

	idx = rtree.index.Index()

	for i, polygon in enumerate(polygons):
		try:
			idx.insert(i, polygon.bounds)
		except:
			logging.exception(
				"rtree error with %s, %s, %s" % (
					polygon.bounds, polygon, polygon.geom_type))
			raise

	remaining = list(range(len(polygons)))
	next_polygon_id = len(polygons)

	candidates = dict((i, polygon) for i, polygon in enumerate(polygons))

	while remaining:
		i = remaining.pop()

		if i not in candidates:
			continue
		indices = [i]
		polygon1 = candidates[i]

		for j in list(idx.intersection(polygon1.bounds)):
			if j != i:
				polygon2 = candidates[j]
				if polygon2.intersects(polygon1):
					indices.append(j)

		if len(indices) > 1:
			merged = shapely.ops.cascaded_union(
				[candidates[k] for k in indices]).convex_hull

			for k in indices:
				idx.delete(k, candidates[k].bounds)
				del candidates[k]

			candidates[next_polygon_id] = merged
			remaining.append(next_polygon_id)
			idx.insert(next_polygon_id, merged.bounds)

			next_polygon_id += 1

	return list(candidates.values())

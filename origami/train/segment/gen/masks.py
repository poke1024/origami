import numpy as np
import cv2
import shapely.geometry

from origami.core.contours import find_contours
from origami.core.mask import Mask


def mask_to_contours(mask, eps_area=100, simplify=3, convex_hulls=True, cls=shapely.geometry.LinearRing):
	mask = mask.astype(np.uint8)

	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	#gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

	contours = find_contours(mask)

	if False:
		image = np.zeros((*mask.shape, 1), dtype=np.uint8)
		image[mask > 0] = 255
		# image = np.broadcast_to(image, [*mask.shape, 3])
		image = np.tile(image, (1, 1, 3))

	# image = np.array(image)
	# print(image.shape, image.dtype)

	polylines = []
	for c in contours:
		if len(c) < 3:
			continue

		if convex_hulls:
			hull = cv2.convexHull(c, returnPoints=False)
			# hull = np.array(hull, dtype=np.int32)
			hull = hull.reshape((hull.shape[0],))
			pts = c.reshape(c.shape[0], 2)[hull]
		else:
			pts = c.reshape(c.shape[0], 2)

		polyline = cls(pts)

		minx, miny, maxx, maxy = polyline.bounds
		if (maxx - minx) * (maxy - miny) < eps_area:
			continue
		area = shapely.geometry.Polygon(pts).area
		#print(area)
		if area < eps_area:
			continue

		polyline = polyline.simplify(simplify, preserve_topology=False)

		if not polyline.is_empty:
			polylines.append(polyline)

		if False:
			pts = polyline.coords
			for a, b in zip(pts, pts[1:]):
				a = np.array(a, dtype=np.int32)
				b = np.array(b, dtype=np.int32)
				cv2.line(image, tuple(a), tuple(b), (200, 0, 0), thickness=4)

	return polylines


def mask_to_polygons(mask, **kwargs):
	return mask_to_contours(mask, cls=shapely.geometry.Polygon, **kwargs)


def polygons_to_mask(shape, polygons):
	assert type(polygons) is list
	h, w = shape
	mask = Mask(shapely.geometry.MultiPolygon(polygons), (0, 0, w, h))
	return mask.binary

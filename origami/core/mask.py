import PIL
import numpy as np

from origami.core.canvas import Canvas


class Mask:
	def __init__(self, shape, bounds=None, buffer=0):
		if bounds is None:
			minx, miny, maxx, maxy = shape.bounds
			minx, miny = np.floor([minx, miny]).astype(np.int32)
			maxx, maxy = np.ceil([maxx, maxy]).astype(np.int32)
			if buffer > 0:
				minx -= buffer
				miny -= buffer
				maxx += buffer
				maxy += buffer
		else:
			minx, miny, maxx, maxy = bounds

		w = int(maxx - minx)
		h = int(maxy - miny)

		canvas = Canvas(w, h)

		try:
			if shape.geom_type == 'Polygon':
				polygons = [shape]
			elif shape.geom_type == 'MultiPolygon':
				polygons = shape.geoms
			else:
				raise ValueError("unsupported shape for mask: %s" % shape.geom_type)

			canvas.set_color(1, 1, 1)

			for polygon in polygons:
				pts = np.array(polygon.exterior.coords) - np.array([minx, miny])
				pts = np.round(pts).astype(np.int32)
				canvas.fill_polygon(pts)

			data = canvas.channel("R")

			self._mask = data > 0
			self._bbox = (minx, miny, w, h)

		finally:
			canvas.finish()

	@property
	def binary(self):
		return self._mask

	@property
	def bounds(self):
		minx, miny, w, h = self._bbox
		return minx, miny, minx + w, miny + h

	def _extract(self, bbox, pixels, background=255):
		x, y, w, h = bbox
		ph, pw = pixels.shape[:2]

		tx = max(x, 0)
		ty = max(y, 0)
		sx = tx - x
		sy = ty - y

		tw = min(w - sx, pw - tx)
		th = min(h - sy, ph - ty)

		cutout = pixels[ty:ty + th, tx:tx + tw].copy()
		assert cutout.shape[:2] == (th, tw)
		if background is not None:
			cutout[np.logical_not(self._mask[sy:sy + th, sx:sx + tw])] = background
		return cutout, (tx, ty)

	def extract(self, pixels, background=255):
		return self._extract(self._bbox, pixels, background)

	def cutout(self, pixels, background=255):
		r, _ = self._extract(self._bbox, pixels, background)
		return r

	def extract_image(self, pixels, background=255):
		cutout, pos = self.extract(pixels, background)
		return PIL.Image.fromarray(cutout), pos

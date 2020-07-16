import numpy as np
import cairocffi as cairo


class Canvas:
	_channels_indices = dict(
		R=2,
		G=1,
		B=0,
		A=3)

	def __init__(self, width, height):
		self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
		self._ctx = cairo.Context(self._surface)
		self._size = (width, height)

	def set_color(self, r, g, b):
		self._ctx.set_source_rgb(r, g, b)

	def fill_polygon(self, pts):
		ctx = self._ctx
		ctx.move_to(*pts[0])
		for p in pts[1:]:
			ctx.line_to(*p)
		ctx.fill()

	def finish(self):
		self._surface.finish()

	def _pixels(self):
		self._surface.flush()

		return np.ndarray(
			shape=(self._surface.get_height(), self._surface.get_stride() // 4),
			dtype=np.uint32,
			buffer=self._surface.get_data())

	def _channels(self, s):
		width, height = self._size
		pixels = self._pixels().view(np.uint8)
		pixels = pixels.reshape(height, width, 4)
		return pixels[:, :, s]

	def pixels(self):
		return self._channels([2, 1, 0, 3])

	def channel(self, name="R"):
		return self._channels(Canvas._channels_indices[name.upper()])

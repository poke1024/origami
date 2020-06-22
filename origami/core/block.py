import numpy as np
import shapely
import math
import cv2
import skimage
import skimage.filters
import PIL.Image
import shapely.wkt
import logging
import multiprocessing.pool

from cached_property import cached_property
from functools import lru_cache

from origami.core.mask import Mask
from origami.core.math import to_shapely_matrix

BACKGROUND = 0.8


def binarize(im, window_size=15):
	cutout = np.array(im)

	if window_size is None:
		window_size = target_height // 2 - 1
	if window_size <= 0:
		try:
			thresh = skimage.filters.threshold_otsu(cutout)
		except ValueError:
			thresh = 128
	else:
		thresh = skimage.filters.threshold_sauvola(cutout, window_size=window_size)
	cutout = (cutout > thresh).astype(np.uint8) * 255
	return PIL.Image.fromarray(cutout)


class Line:
	def __init__(self, block, p, right, up, tesseract_data, wkt=None, text_area=None):
		self._tesseract_data = tesseract_data
		self._block = block

		self._p = np.array(p)
		self._right = np.array(right)
		self._up = np.array(up)

		if wkt:
			self._polygon = shapely.wkt.loads(wkt)
		else:
			self._polygon = text_area.intersection(shapely.geometry.Polygon([
				self._p, self._p + self._right, self._p + self._right + self._up, self._p + self._up])).convex_hull

		self._text = ""

	@property
	def block(self):
		return self._block

	@property
	def center(self):
		return self._p + self._right / 2

	@property
	def angle(self):
		return math.atan2(self._right[1], self._right[0])

	@property
	def text(self):
		return self._text

	def set_text(self, text):
		self._text = text

	def image(self, target_height=48, dewarped=True, deskewed=True, binarized=False, window_size=15):
		if dewarped:
			im = self.dewarped_image(target_height)
		elif deskewed:
			im = self.deskewed_image(target_height)
		else:
			im = self.warped_image
		if binarized:
			im = binarize(im, window_size)
		return im

	@property
	def warped_image(self):
		mask = Mask(self.image_space_polygon)
		image, pos = mask.extract_image(
			self._block.page_pixels,
			background=self._block.background)
		return image

	def deskewed_image(self, target_height=48, interpolation=cv2.INTER_AREA):
		p, right, up = self._p, self._right, self._up
		width = int(math.ceil(np.linalg.norm(right)))

		matrix = cv2.getAffineTransform(
			np.array([p, p + right, p + up]).astype(np.float32),
			np.array([
				(0, target_height - 1),
				(width, target_height - 1),
				(0, 0)]).astype(np.float32))

		pixels = self._block.page_pixels

		warped = cv2.warpAffine(
			pixels, matrix, (width, target_height), interpolation,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=self._block.background)

		try:
			mask = Mask(
				shapely.affinity.affine_transform(
					self._polygon, to_shapely_matrix(matrix)),
				bounds=(0, 0, width, target_height))

			background = np.quantile(warped, BACKGROUND)
			cutout = mask.cutout(warped, background=background)

		except ValueError:
			# might happen on unsupported mask geometry types.
			cutout = warped

		return PIL.Image.fromarray(cutout)

	def dewarped_image(self, target_height=48, interpolation=cv2.INTER_LINEAR):
		assert self.block.dewarped
		p0 = self._p
		right = self._right
		up = self._up

		ys = np.linspace([0, 0], up, target_height)
		xs = np.linspace([0, 0], right, int(np.ceil(np.linalg.norm(right))))

		dewarped_grid = (ys + p0)[:, np.newaxis] + xs[np.newaxis, :]
		dewarped_grid = np.flip(dewarped_grid, axis=-1)
		inv = self.block.page.dewarper.grid.inverse
		warped_grid = inv(dewarped_grid.reshape((len(ys) * len(xs), 2)))
		warped_grid = warped_grid.reshape((len(ys), len(xs), 2)).astype(np.float32)

		pixels = np.array(self.block.page.warped)
		pixels = cv2.remap(pixels, warped_grid, None, interpolation)
		pixels = pixels[::-1, :]

		return PIL.Image.fromarray(pixels)

	@property
	def coords(self):
		try:
			return list(self.image_space_polygon.exterior.coords)
		except:
			return list(self.image_space_polygon.convex_hull.exterior.coords)

	@property
	def image_space_polygon(self):
		return self._polygon

	@property
	def info(self):
		return dict(
			p=self._p.tolist(),
			right=self._right.tolist(),
			up=self._up.tolist(),
			wkt=self._polygon.wkt,
			tesseract_data=dict(
				baseline=self._tesseract_data['baseline'],
				descent=self._tesseract_data['descent'],
				ascent=self._tesseract_data['ascent'],
				height=self._tesseract_data['height']))

	@property
	def length(self):
		return np.linalg.norm(self._right)


def _extended_baseline(text_area, p, right, up, max_ext=3):
	minx, miny, maxx, maxy = text_area.bounds
	magnitude = max(maxx - minx, maxy - miny)
	u = (right / np.linalg.norm(right)) * 2 * magnitude
	line = shapely.geometry.LineString(
		[p - u, p + u]).intersection(text_area)
	if line.geom_type != "LineString":
		print("failed to find extended baseline")
	else:
		coords = list(line.coords)
		if len(coords) >= 2:
			xp = np.array(min(coords, key=lambda xy: xy[0]))
			xq = np.array(max(coords, key=lambda xy: xy[0]))

			extra = 0
			if (xp - p).dot(right) < 0:
				extra = np.linalg.norm(xp - p)
				right = (p + right) - xp
				p = xp

			old_length = np.linalg.norm(right)
			new_length = min(np.linalg.norm(xq - p), extra + old_length * max_ext)

			if new_length > old_length:
				right = right * (new_length / old_length)

	return dict(p=p, right=right, up=up)


class Block:
	def __init__(self, page, polygon, dewarped):
		self._image_space_polygon = polygon
		self._page = page
		self._dewarped = dewarped

	@property
	def page(self):
		return self._page
	
	@property
	def page_pixels(self):
		return self.page.pixels(self._dewarped)
		
	@property
	def dewarped(self):
		return self._dewarped

	@cached_property
	def image(self):
		im, _ = self.extract_image()
		return im

	@lru_cache(maxsize=3)
	def extract_image(self, buffer=10):
		mask = Mask(self.text_area(buffer=buffer))
		return mask.extract_image(
			self.page_pixels, background=self.background)

	@property
	def image_space_polygon(self):
		return self._image_space_polygon

	def text_area(self, buffer=10, fringe_limit=0.1):
		# fringe_limit is currently ignored. this used to have a bbox fallback,
		# which was a very bad idea, since it was not aware of skew.
		return self.image_space_polygon.buffer(buffer).convex_hull

	@property
	def coords(self):
		return list(self._image_space_polygon.exterior.coords)

	@property
	def pos(self):
		return np.array(self._pos)

	@cached_property
	def background(self):
		mask = Mask(self.image_space_polygon)
		im, _ = mask.extract_image(self.page_pixels, background=None)
		pixels = np.array(im)
		return np.quantile(pixels, BACKGROUND)

	@property
	def _extent(self):
		minx, miny, maxx, maxy = self.image_space_polygon.bounds
		return max(maxx - minx, maxy - miny)


def padded(im, pad=32, background_color=255):
	width, height = im.size
	result = PIL.Image.new(im.mode, (width, height + 2 * pad), background_color)
	result.paste(im, (0, pad))
	return result


class LineDetector:
	def __init__(
		self,
		force_parallel_lines=False,
		fringe_limit=0.1,
		extra_height=0.05,
		extra_descent=0,
		text_buffer=10):

		self._force_parallel_baselines = force_parallel_lines
		self._fringe_limit = fringe_limit
		self._extra_height = extra_height
		self._extra_descent = extra_descent
		self._text_buffer = text_buffer

	def detect_baselines(self, block):
		import tesserocr

		with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:

			api.SetVariable(
				"textord_parallel_baselines",
				"1" if self._force_parallel_baselines else "0")
			api.SetVariable("textord_straight_baselines", "1")

			# without padding, Tesseract sometimes underestimates row heights
			# for single line headers.

			pad = 32
			im, pos = block.extract_image(self._text_buffer)
			api.SetImage(padded(im, pad=pad))
			pos = np.array(pos) - np.array([0, pad])

			api.AnalyseLayout()
			ri = api.GetIterator()

			baselines = []
			if ri:
				# if no lines are available, api.AnalyseLayout() will produce None as iterator.

				level = tesserocr.RIL.TEXTLINE
				for r in tesserocr.iterate_level(ri, level):
					baseline = r.Baseline(level)
					if not baseline:
						continue

					p1, p2 = baseline
					attr = r.RowAttributes()

					# note: row_height = x_height + ascenders + abs(descenders)
					# as defined in LTRResultIterator::RowAttributes, see
					# https://github.com/tesseract-ocr/tesseract/blob/
					# acc4c8bff55c4dfdbec088fe2c507285fa7c2e27/src/ccmain/ltrresultiterator.cpp#L148

					baselines.append(dict(
						baseline=(
							tuple((np.array(p1) + pos).tolist()),
							tuple((np.array(p2) + pos).tolist())),
						descent=attr['descenders'],  # always negative
						ascent=attr['ascenders'],  # always positive
						height=attr['row_height']))

		return baselines

	def detect_lines(self, block):
		text_area = block.text_area(
			fringe_limit=self._fringe_limit,
			buffer=self._text_buffer)
		lines = []
		for baseline in self.detect_baselines(block):
			p1, p2 = baseline['baseline']

			# Tesseract tends to underestimate row height. work
			# around by adding another few percent.
			height = baseline['height'] * (1 + self._extra_height)
			descent = baseline['descent'] * (1 + self._extra_descent)

			right = (np.array(p2) - np.array(p1)).astype(np.float64)

			up = -np.array([-right[1], right[0]])
			up /= np.linalg.norm(up)
			down = -up

			lines.append(
				Line(
					block,
					**_extended_baseline(
						text_area,
						p=np.array(p1) + abs(descent) * down,
						right=right,
						up=up * height),
					tesseract_data=baseline,
					text_area=text_area))

		return lines


class ConcurrentLineDetector:
	def __init__(self, processes=8, **kwargs):
		self._detector = LineDetector(**kwargs)
		self._pool = multiprocessing.pool.ThreadPool(processes=processes)

	def _detect_lines(self, item):
		block_path, block = item

		try:
			return block_path, self._detector.detect_lines(block)
		except:
			logging.error("failed to detect lines on block %s" % str(block_path))
			raise

	def __call__(self, blocks):
		return dict(self._pool.map(self._detect_lines, blocks.items()))

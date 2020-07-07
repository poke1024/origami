import numpy as np
import shapely
import math
import cv2
import skimage
import skimage.filters
import scipy.spatial
import PIL.Image
import shapely.wkt
import logging
import traceback
import multiprocessing.pool
import enum

from cached_property import cached_property
from functools import lru_cache

from origami.core.mask import Mask
from origami.core.math import to_shapely_matrix
from origami.core.binarize import Binarizer


BACKGROUND = 0.8
DEFAULT_BUFFER = 0.0015


def intersect_segments(a, b, default=None):
	c = a.intersection(b)
	if c.geom_type == "Point":
		return np.array(list(c.coords)[0])
	else:
		return default


class Line:
	def __init__(
		self, block, p, right, up,
		tesseract_data,
		wkt=None, text_area=None, confidence=1):

		self._tesseract_data = tesseract_data
		self._block = block

		self._p = np.array(p)
		self._right = np.array(right)
		self._up = np.array(up)

		if wkt:
			self._polygon = shapely.wkt.loads(wkt)
		else:
			self._polygon = text_area.intersection(shapely.geometry.Polygon([
				self._p, self._p + self._right,
				self._p + self._right + self._up, self._p + self._up])).convex_hull

		self._confidence = confidence

	@property
	def block(self):
		return self._block

	@property
	def center(self):
		p1, p2 = self._tesseract_data['baseline']
		return (np.array(p1) + np.array(p2)) / 2

	@property
	def angle(self):
		return math.atan2(self._right[1], self._right[0])

	@property
	def confidence(self):
		return self._confidence

	def update_confidence(self, confidence):
		self._confidence = confidence

	def annotate(self, text_area):
		im, pos = self.block.extract_image(buffer)
		pixels = np.array(im.convert("RGB"))

		pts = np.array(list(text_area.exterior.coords)) - pos
		for p, q in zip(pts, pts[1:]):
			cv2.line(pixels, tuple(map(int, p)), tuple(map(int, q)), (0, 255, 0), 2)

		ex = _extended_baseline(text_area, self._p, self._right, self._up)
		ex_p1 = np.array(ex["p"]) - pos
		ex_p2 = ex_p1 + np.array(ex["right"])
		ex_p1 = tuple(map(int, np.array(ex_p1)))
		ex_p2 = tuple(map(int, np.array(ex_p2)))
		cv2.line(pixels, ex_p1, ex_p2, (0, 0, 255), 3)

		p1, p2 = self._tesseract_data['baseline']
		p1 = tuple(map(int, np.array(p1) - pos))
		p2 = tuple(map(int, np.array(p2) - pos))
		cv2.line(pixels, p1, p2, (255, 0, 0), 2)

		return PIL.Image.fromarray(pixels)

	def image(
		self, target_height=48, column=None,
		dewarped=True, deskewed=True, binarizer=None):

		if dewarped:
			im = self.dewarped_image(target_height, column=column)
		elif deskewed:
			im = self.deskewed_image(target_height)
		else:
			im = self.warped_image

		if binarizer:
			im = binarizer(im)

		return im

	def masked_image(self, mode="polygon"):
		if mode not in ("polygon", "bbox"):
			raise ValueError(mode)
		mask = Mask(self.image_space_polygon)
		bg = self._block.background if mode == "polygon" else None
		image, pos = mask.extract_image(
			self._block.page_pixels,
			background=bg)
		return image

	def _extract_deskewed(
		self, pixels, target_height=48,
		background=255, interpolation=cv2.INTER_AREA):

		p, right, up = self._p, self._right, self._up
		width = int(math.ceil(np.linalg.norm(right)))

		matrix = cv2.getAffineTransform(
			np.array([p, p + right, p + up]).astype(np.float32),
			np.array([
				(0, target_height - 1),
				(width, target_height - 1),
				(0, 0)]).astype(np.float32))

		warped = cv2.warpAffine(
			pixels, matrix, (width, target_height), interpolation,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=background)

		try:
			mask = Mask(
				shapely.affinity.affine_transform(
					self._polygon, to_shapely_matrix(matrix)),
				bounds=(0, 0, width, target_height))

			cutout = mask.cutout(warped, background=background)

		except ValueError:
			# might happen on unsupported mask geometry types.
			cutout = warped

		return cutout

	def deskewed_image(self, target_height=48, interpolation=cv2.INTER_AREA):
		return PIL.Image.fromarray(self._extract_deskewed(
			self._block.page_pixels,
			target_height,
			self._block.background,
			interpolation))

	def _position(self, xres, column):
		p0 = self._p
		right = self._right
		up = self._up

		if column is not None:
			p1 = p0 + right
			px0, py0, px1, py1 = shapely.geometry.LineString([p0, p1]).bounds

			x0, x1 = column
			if x0 is None:
				x0 = px0
			if x1 is None:
				x1 = px1

			s0 = shapely.geometry.LineString([[x0, py0 - 1], [x0, py1 + 1]])
			s1 = shapely.geometry.LineString([[x1, py0 - 1], [x1, py1 + 1]])

			bottom = shapely.geometry.LineString([p0, p1])

			p0 = intersect_segments(bottom, s0, default=p0)
			p1 = intersect_segments(bottom, s1, default=p1)

			right = p1 - p0
			xres *= (x1 - x0) / (px1 - px0)

		return p0, right, up, xres

	def warped_grid(self, xsteps=None, ysteps=None, xres=1, yres=1, column=None):
		p0, right, up, xres = self._position(xres, column)

		if xsteps is None or ysteps is None:
			rough_grid = self.warped_grid(xsteps=2, ysteps=2)
			assert tuple(rough_grid.shape[:2]) == (2, 2)

		if xsteps is None:
			xsteps = np.max(np.abs(rough_grid[:, 0, 0] - rough_grid[:, 1, 0]))
			xsteps = max(2, int(np.ceil(xsteps * xres)))

		if ysteps is None:
			ysteps = np.max(np.abs(rough_grid[0, :, 1] - rough_grid[1, :, 1]))
			ysteps = max(2, int(np.ceil(ysteps * yres)))

		ys = np.linspace([0, 0], up, ysteps)
		xs = np.linspace([0, 0], right, xsteps)

		dewarped_grid = (ys + p0)[:, np.newaxis] + xs[np.newaxis, :]
		dewarped_grid = np.flip(dewarped_grid, axis=-1)
		inv = self.block.page.dewarper.grid.inverse
		warped_grid = inv(dewarped_grid.reshape((len(ys) * len(xs), 2)))
		warped_grid = warped_grid.reshape((len(ys), len(xs), 2)).astype(np.float32)

		# grid is [y, x, p] and p is [x, y].
		return warped_grid

	def dewarped_image(self, target_height=48, column=None, interpolation=cv2.INTER_LINEAR):
		assert self.block.stage.is_dewarped

		warped_grid = self.warped_grid(ysteps=target_height, column=column)

		pixels = np.array(self.block.page.warped)
		pixels = cv2.remap(pixels, warped_grid, None, interpolation)
		pixels = pixels[::-1, :]

		return PIL.Image.fromarray(pixels)

	def warped_path(self, resolution=1):
		assert self.block.stage.is_dewarped

		warped_grid = self.warped_grid(ysteps=2, xres=resolution)

		height = np.median(np.linalg.norm(warped_grid[1] - warped_grid[0], axis=-1))
		return np.mean(warped_grid, axis=0), abs(height)

	@cached_property
	def ink(self):
		assert self._block.stage == Stage.WARPED

		p, right, up = self._p, self._right, self._up
		height = int(math.ceil(np.linalg.norm(up)))

		cutout = self._extract_deskewed(
			np.array(self._block.page.binarized),
			height,
			255,
			cv2.INTER_AREA)

		return 1 - np.mean(cutout.astype(np.float32) / 255, axis=0)

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
			confidence=self._confidence,
			tesseract_data=dict(
				baseline=self._tesseract_data['baseline'],
				descent=self._tesseract_data['descent'],
				ascent=self._tesseract_data['ascent'],
				height=self._tesseract_data['height']))

	@cached_property
	def length(self):
		return np.linalg.norm(self._right)

	@cached_property
	def unextended_length(self):
		p1, p2 = self._tesseract_data['baseline']
		return np.linalg.norm(np.array(p1) - np.array(p2))

	@cached_property
	def height(self):
		return np.linalg.norm(self._up)

	def dewarped_height(self, dewarper):
		assert not self._block.stage.is_dewarped
		p0, right, up = self._p, self._right, self._up
		tfm = dewarper.grid.transformer
		p1 = p0 + up
		tx, ty = tfm(*np.array([p0, p1]).transpose())
		q0, q1 = np.array([tx, ty]).transpose()
		return np.linalg.norm(q1 - q0)


def _extended_baseline(text_area, p, right, up, max_ext=3):
	coords = []

	for retry in range(2):
		minx, miny, maxx, maxy = text_area.bounds
		magnitude = max(maxx - minx, maxy - miny)
		u = (right / np.linalg.norm(right)) * 2 * magnitude
		line = shapely.geometry.LineString(
			[p - u, p + u]).intersection(text_area)
		if line.geom_type == "LineString":
			coords = list(line.coords)
			break
		if retry == 0:
			text_area = text_area.convex_hull

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
	else:
		logging.warning("no extended baseline for (%s, %s, %s) in area %s" % (
			p, right, up, text_area.bounds))

	return dict(p=p, right=right, up=up)


class Block:
	def __init__(self, page, polygon, stage):
		self._image_space_polygon = polygon
		self._page = page
		self._stage = stage

	@property
	def page(self):
		return self._page
	
	@property
	def page_pixels(self):
		return self.page.pixels(self._stage.is_dewarped)
		
	@property
	def stage(self):
		return self._stage

	@property
	def is_empty(self):
		return self._image_space_polygon.is_empty

	def image(self, text_area):
		mask = Mask(text_area)
		return mask.extract_image(
			self.page_pixels, background=self.background)

	@property
	def image_space_polygon(self):
		return self._image_space_polygon

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


class TextAreaFactory:
	def __init__(self, blocks=[], buffer=DEFAULT_BUFFER):
		self._blocks = blocks
		self._buffer = buffer
		self._tree = shapely.strtree.STRtree([
			block.image_space_polygon for block in blocks
		])

	def __call__(self, block):
		buffer = block.page.geometry(
			block.stage.is_dewarped).rel_length(self._buffer)
		polygon = block.image_space_polygon.buffer(buffer)
		for other in self._tree.query(polygon):
			if other is not block.image_space_polygon:
				polygon = polygon.difference(other)
		return polygon


def padded(im, pad=32, background_color=255):
	im = im.convert("L")
	width, height = im.size
	result = PIL.Image.new(
		im.mode,
		(width, height + 2 * pad),
		int(background_color))
	result.paste(im, (0, pad))
	return result


class LineDetector:
	def __init__(
		self,
		force_parallel_lines=False,
		force_lines=False,
		extra_height=0.05,
		extra_descent=0,
		block_size_minimum=4,
		text_area_factory=TextAreaFactory(),
		binarizer=Binarizer()):

		self._force_parallel_baselines = force_parallel_lines
		self._force_lines = force_lines

		self._extra_height = extra_height
		self._extra_descent = extra_descent

		self._text_area_factory = text_area_factory
		self._binarizer = binarizer
		self._block_size_minimum = block_size_minimum

	def create_fake_line(self, block, text_area):
		minx, miny, maxx, maxy = block.image_space_polygon.bounds
		h = maxy - miny

		p1 = np.array([minx, maxy])
		p2 = np.array([maxx, maxy])
		up = np.array([0, -h])

		baseline = dict(
			baseline=(p1.tolist(), p2.tolist()),
			descent=0,
			ascent=h,
			height=h)

		return Line(
			block,
			p=p1, right=p2 - p1, up=up,
			tesseract_data=baseline,
			text_area=text_area)

	def detect_baselines(self, block, text_area):
		import tesserocr

		with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:

			api.SetVariable(
				"textord_parallel_baselines",
				"1" if self._force_parallel_baselines else "0")
			api.SetVariable("textord_straight_baselines", "1")

			# without padding, Tesseract sometimes underestimates row heights
			# for single line headers or does not recoginize header lines at all.

			if self._binarizer is not None:
				bg = 255
			else:
				bg = block.background

			pad = 32
			im, pos = block.image(text_area)
			if min(im.width, im.height) < self._block_size_minimum:
				return []
			im = padded(im, pad=pad, background_color=bg)

			if self._binarizer is not None:
				# binarizing Tesseract's input detects some correct baselines
				# that are omitted on grayscale input.
				im = self._binarizer(im)

			api.SetImage(im)
			pos = np.array(pos) - np.array([0, pad])

			api.AnalyseLayout()
			ri = api.GetIterator()

			baselines = []
			if ri:
				# if no lines are available, api.AnalyseLayout() will produce
				# None as iterator.

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
		if block.is_empty:
			return []

		text_area = self._text_area_factory(block)
		if text_area.is_empty:
			return []

		lines = []
		for baseline in self.detect_baselines(block, text_area):
			p1, p2 = baseline['baseline']
			descent = baseline['descent']

			# Tesseract tends to underestimate row height. work
			# around by adding another few percent.
			height = baseline['height'] * (1 + self._extra_height)

			right = (np.array(p2) - np.array(p1)).astype(np.float64)

			up = -np.array([-right[1], right[0]])
			up /= np.linalg.norm(up)
			down = -up

			spec = _extended_baseline(
				text_area,
				p=np.array(p1, dtype=np.float64),
				right=right,
				up=up * height)

			x_descent = abs(descent * (1 + self._extra_descent))
			spec["p"] += x_descent * down.astype(np.float64)

			lines.append(
				Line(
					block,
					**spec,
					tesseract_data=baseline,
					text_area=text_area))

		if self._force_lines and not lines:
			lines.append(self.create_fake_line(block, text_area))

		return lines


class ConcurrentLineDetector:
	def __init__(self, processes=8, **kwargs):
		self._detector = LineDetector(**kwargs)
		self._processes = processes

	def _detect_lines(self, item):
		block_path, block = item

		try:
			return block_path, self._detector.detect_lines(block)
		except:
			logging.error("failed to detect lines on block %s" % str(block_path))
			logging.error(traceback.format_exc())
			raise

	def __call__(self, blocks):
		with multiprocessing.pool.ThreadPool(processes=self._processes) as pool:
			return dict(pool.map(self._detect_lines, blocks.items()))

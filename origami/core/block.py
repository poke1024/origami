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

from cached_property import cached_property
from functools import lru_cache

from origami.core.mask import Mask
from origami.core.math import to_shapely_matrix
from origami.concaveman import concaveman2d


BACKGROUND = 0.8
DEFAULT_BUFFER = 0.0015


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
	def __init__(self, block, p, right, up, contour_data, tesseract_data, wkt=None, text_area=None):
		self._contour_data = contour_data
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

	@property
	def block(self):
		return self._block

	@property
	def center(self):
		return self._p + self._right / 2

	@property
	def angle(self):
		return math.atan2(self._right[1], self._right[0])

	def annotate(self, buffer=DEFAULT_BUFFER):
		im, pos = self.block.extract_image(buffer)
		pixels = np.array(im.convert("RGB"))

		text_area = self.block.text_area(**self._contour_data)

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

	def masked_image(self, mode="polygon"):
		if mode not in ("polygon", "bbox"):
			raise ValueError(mode)
		mask = Mask(self.image_space_polygon)
		bg = self._block.background if mode == "polygon" else None
		image, pos = mask.extract_image(
			self._block.page_pixels,
			background=bg)
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

	def warped_path(self, resolution=1):
		assert self.block.dewarped

		p0 = self._p
		right = self._right
		up = self._up

		ys = [[0, 0], up]
		xs = np.linspace([0, 0], right, int(
			np.ceil(resolution * np.linalg.norm(right))))

		dewarped_grid = (ys + p0)[:, np.newaxis] + xs[np.newaxis, :]
		dewarped_grid = np.flip(dewarped_grid, axis=-1)
		inv = self.block.page.dewarper.grid.inverse
		warped_grid = inv(dewarped_grid.reshape((len(ys) * len(xs), 2)))
		warped_grid = warped_grid.reshape((len(ys), len(xs), 2)).astype(np.float32)

		height = np.median(np.linalg.norm(warped_grid[1] - warped_grid[0], axis=-1))
		return np.mean(warped_grid, axis=0), abs(height)

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
			contour_data=dict(
				buffer=self._contour_data["buffer"],
				concavity=self._contour_data["concavity"],
				detail=self._contour_data["detail"]
			),
			tesseract_data=dict(
				baseline=self._tesseract_data['baseline'],
				descent=self._tesseract_data['descent'],
				ascent=self._tesseract_data['ascent'],
				height=self._tesseract_data['height']))

	@property
	def length(self):
		return np.linalg.norm(self._right)

	@property
	def unextended_length(self):
		p1, p2 = self._tesseract_data['baseline']
		return np.linalg.norm(np.array(p1) - np.array(p2))


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
		logging.error("no extended baseline for (%s, %s, %s) in area %s" % (
			p, right, up, text_area.bounds))

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

	@lru_cache(maxsize=3)
	def image(self, **kwargs):
		mask = Mask(self.text_area(**kwargs))
		return mask.extract_image(
			self.page_pixels, background=self.background)

	@property
	def image_space_polygon(self):
		return self._image_space_polygon

	@lru_cache(maxsize=3)
	def text_area(self, buffer=DEFAULT_BUFFER, concavity=2, detail=0.01):
		mag = self.page.magnitude(self._dewarped)

		poly = self.image_space_polygon.buffer(mag * buffer)

		if concavity > 0:
			ext = np.array(poly.exterior.coords)
			pts = concaveman2d(
				ext,
				scipy.spatial.ConvexHull(ext).vertices,
				concavity=concavity,
				lengthThreshold=mag * detail)
		else:  # disable concaveman
			pts = list(poly.convex_hull.exterior.coords)

		return shapely.geometry.Polygon(pts)

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
		extra_height=0.05,
		extra_descent=0,
		contours_buffer=DEFAULT_BUFFER,
		contours_concavity=2,
		contours_detail=0.001,
		binarize=binarize):

		self._force_parallel_baselines = force_parallel_lines

		self._extra_height = extra_height
		self._extra_descent = extra_descent

		self._contour_data = dict(
			buffer=contours_buffer,
			concavity=contours_concavity,
			detail=contours_detail)

		self._binarize = binarize

	def detect_baselines(self, block):
		import tesserocr

		with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_BLOCK) as api:

			api.SetVariable(
				"textord_parallel_baselines",
				"1" if self._force_parallel_baselines else "0")
			api.SetVariable("textord_straight_baselines", "1")

			# without padding, Tesseract sometimes underestimates row heights
			# for single line headers or does not recoginize header lines at all.

			if self._binarize is not None:
				bg = 255
			else:
				bg = block.background

			pad = 32
			im, pos = block.image(**self._contour_data)
			im = padded(im, pad=pad, background_color=bg)

			if self._binarize is not None:
				# binarizing Tesseract's input detects some correct baselines
				# that are omitted on grayscale input.
				im = self._binarize(im)

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
		text_area = block.text_area(**self._contour_data)
		lines = []
		for baseline in self.detect_baselines(block):
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
					contour_data=self._contour_data,
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
			logging.error(traceback.format_exc())
			raise

	def __call__(self, blocks):
		return dict(self._pool.map(self._detect_lines, blocks.items()))

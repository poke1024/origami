#!/usr/bin/env python3

import click
import json
import numpy as np
import logging
import math
import skimage
import cv2
import io
import scipy
import shapely
import PIL.Image
import skimage.filters
import skimage.morphology

from pathlib import Path
from sklearn.decomposition import PCA
from cached_property import cached_property

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.dewarp import Samples
from origami.core.block import ConcurrentLineDetector
from origami.core.math import Geometry, divide_path
from origami.core.mask import Mask
from origami.batch.core.utils import RegionsFilter


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
	def __init__(self, line_det, max_phi_rad, min_length=50, eccentricity=0.99):
		self._line_detector = line_det
		self._max_phi = max_phi_rad
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


def _angles(samples, coords, max_segment=0.05):
	coords = np.array(coords)

	# normalize. need to check against direction here.
	# if coords[0, 1] > coords[-1, 1]:
	#	coords = coords[::-1]

	coords = divide_path(
		coords, samples.geometry.rel_length(max_segment))

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


def _parse_sep(names):
	return tuple(map(lambda t: t.strip(), names.split(",")))


class FlowDetectionProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.CONTOURS, stage=Stage.WARPED)),
			("output", Output(Artifact.FLOW, Artifact.LINES, stage=Stage.WARPED))
		]

	def add_line_skew_hq(self, samples, blocks, lines, max_phi_rad, delta=0):
		n_skipped = 0
		for line in lines.values():
			if abs(line.angle) < max_phi_rad:
				samples.append(line.center, line.angle + delta)
			else:
				n_skipped += 1
		#if n_skipped > 0:
		#	logging.warning("skipped %d lines." % n_skipped)

	def add_line_skew_lq(self, samples, blocks, lines, max_phi_rad):
		estimator = LineSkewEstimator(
			line_det=SobelLineDetector(),
			max_phi_rad=max_phi_rad)

		def add(im, pos):
			for pt, phi in estimator(im):
				samples.append(np.array(pt) + np.array(pos), phi)

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

	def add_separator_skew(self, samples, separators, sep_types, max_std=0.1):
		for path, polyline in separators.items():
			if path[1] in sep_types:
				sep_points, sep_values = _angles(samples, polyline.coords)

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

				samples.extend(sep_points, sep_values)

	def add_border_skew(self, samples, page, blocks, separators, **kwargs):
		estimator = BorderEstimator(page, blocks, separators)
		for coords in estimator.paths(**kwargs):
			sep_points, sep_values = _angles(samples, coords)
			samples.extend(sep_points, sep_values)

	def process(self, page_path: Path, warped, output):
		detector = ConcurrentLineDetector(
			force_parallel_lines=False,
			extend_baselines=False,
			single_column=False)

		max_phi_rad = self._options["max_phi"] * (math.pi / 180)
		max_std = self._options["max_phi_std"]

		page = warped.page
		blocks = warped.regions.by_path
		block_lines = detector(warped.regions.by_path)

		lines = dict()
		for k, v in block_lines.items():
			for i, line in enumerate(v):
				lines[k + (i,)] = line

		separators = warped.separators.by_path
		rescale_separators = False

		min_length = page.geometry(dewarped=False).rel_length(
			self._options["min_line_length"])

		def filter_geoms(geoms, length):
			return dict(
				(k, g) for k, g in geoms.items()
				if length(g) > min_length)

		lines = filter_geoms(lines, lambda l: l.unextended_length)
		separators = filter_geoms(separators, lambda g: g.length)

		r_filter = RegionsFilter(self._options["regions"])
		lines = dict(
			(k, g)
			for k, g in lines.items()
			if r_filter(k))

		if separators is not None and rescale_separators:  # legacy mode
			sx = self._width / 1280
			sy = self._height / 2400
			separators = dict((k, shapely.affinity.scale(
				s, sx, sy, origin=(0, 0))) for k, s in separators.items())

		geometry = page.geometry(False)
		samples_h = Samples(geometry)
		samples_v = Samples(geometry)

		if separators:
			self.add_separator_skew(
				samples_h, separators,
				_parse_sep(self._options["horizontal_separators"]), max_std=max_std)
			self.add_separator_skew(
				samples_v, separators,
				_parse_sep(self._options["vertical_separators"]), max_std=max_std)

		if lines:
			self.add_line_skew_hq(
				samples_h, blocks, lines, max_phi_rad=max_phi_rad, delta=0)
			self.add_line_skew_hq(
				samples_v, blocks, lines, max_phi_rad=max_phi_rad, delta=math.pi / 2)

		if self._options["estimate_border_skew"]:
			self.add_border_skew(samples_v, page, blocks, separators)

		with output.flow() as zf:
			samples_h.save(zf, "h")
			samples_v.save(zf, "v")

		with output.lines() as zf:
			info = dict(version=1)
			zf.writestr("meta.json", json.dumps(info))

			for parts, lines in block_lines.items():
				prediction_name = parts[0]
				class_name = parts[1]
				block_id = parts[2]

				for line_id, line in enumerate(lines):
					line_name = "%s/%s/%s/%d" % (
						prediction_name, class_name, block_id, line_id)
					zf.writestr("%s.json" % line_name, json.dumps(line.info))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--max-phi',
	type=float,
	default=30,
	help="maximum allowed skewing angle.")
@click.option(
	'--max-phi-std',
	type=float,
	default=0.1,
	help="maximum allowed standard deviation inside angle set.")
@click.option(
	'--min-line-length',
	type=float,
	default=0.05,
	help="detect warp using baselines that are above this relative length.")
@click.option(
	'--regions',
	type=str,
	default="regions/TEXT, regions/TABULAR",
	help="which regions to consider for warping estimation")
@click.option(
	'--horizontal-separators',
	type=str,
	default="H",
	help="which horizontal separator types to use for warping estimation")
@click.option(
	'--vertical-separators',
	type=str,
	default="V, T",
	help="which horizontal separator types to use for warping estimation")
@click.option(
	'--estimate-border-skew',
	is_flag=True,
	default=False,
	help="should the border be detected and estimated?")
@Processor.options
def detect_flow(data_path, **kwargs):
	""" Perform flow and warp detection on all document images in DATA_PATH. Needs
	information from contours batch. """
	processor = FlowDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_flow()


#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.signal
import scipy.spatial
import click
import shapely.ops
import shapely.wkt
import shapely.strtree
import shapely.geometry
import shapely.ops
import sklearn.cluster
import skimage.filters
import skimage.morphology
import networkx as nx
import collections
import portion
import intervaltree
import logging
import importlib
import cv2
import json
import PIL.Image

from pathlib import Path
from functools import partial
from cached_property import cached_property

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.separate import Separators
from origami.core.xycut import polygon_order
from origami.core.neighbors import neighbors
from origami.core.utils import build_func_from_string
from origami.batch.core.utils import RegionsFilter
from origami.core.predict import PredictorType


def overlap_ratio(a, b):
	if a.area > b.area:
		a, b = b, a
	return a.intersection(b).area / a.area


def fixed_point(func, x0, reduce):
	while True:
		x = func(x0)
		if reduce(x) == reduce(x0):
			return x
		x0 = x


def _cohesion(shapes, union):
	return sum([shape.area for shape in shapes]) / union.area


def kernel(*s):
	return np.ones(s) / np.prod(s)


def _line_length(geom):
	if geom.geom_type == "LineString":
		return geom.length
	elif geom.geom_type == "MultiLineString":
		return sum(map(_line_length, geom.geoms))
	else:
		return 0


class LineCounts:
	def __init__(self, lines):
		num_lines = collections.defaultdict(int)
		for path, line in lines.items():
			num_lines[path[:3]] += 1
		self._num_lines = num_lines

	def add(self, name, count):
		self._num_lines[name] = count

	def remove(self, name):
		if name in self._num_lines:
			del self._num_lines[name]

	def combine(self, names, target):
		self._num_lines[target] = sum([
			self._num_lines.get(x, 0) for x in names
		])

	def __getitem__(self, block_path):
		return self._num_lines.get(block_path, 0)


def non_empty_contours(contours):
	for k, contour in contours:
		if not contour.is_empty:
			if not contour.is_valid:
				# this should have happened in the dewarp stage.
				contour = contour.buffer(0)
			yield k, contour


class Regions:
	def __init__(self, page, warped_lines, contours, separators, segmentation):
		self._page = page

		self._contours = dict(non_empty_contours(contours))
		self._unmodified_contours = self._contours.copy()

		self._names = {}
		for k, contour in contours:
			self._names[id(contour)] = "/".join(k)

		self._separators = separators
		self._segmentation = segmentation

		self._line_counts = LineCounts(warped_lines)
		self._warped_lines = warped_lines
		self._union = None
		self._mapped_from = collections.defaultdict(list)

		max_labels = collections.defaultdict(int)
		for k in self._contours.keys():
			max_labels[k[:2]] = max(max_labels[k[:2]], int(k[2]))
		self._max_labels = max_labels

	def debug_save(self, path):
		data = dict()
		for k, contour in self._contours.items():
			data["/".join(k)] = contour.wkt
		with open(path, "w") as f:
			f.write(json.dumps(data))

	def check_geometries(self, allowed):
		for k, contour in self._contours.items():
			if not contour.is_valid:
				raise ValueError("invalid contour")
			if contour.geom_type not in allowed:
				raise ValueError("%s not in %s" % (
					contour.geom_type, allowed))

	def set_union_operator(self, u):
		self._union = u

	@property
	def page(self):
		return self._page

	@property
	def separators(self):
		return self._separators

	@cached_property
	def grayscale(self):
		return np.array(self._page.dewarped.convert("L"))

	@cached_property
	def binarized(self):
		grayscale = self.grayscale

		m_lh = self.median_line_height

		window_size = m_lh // 2
		if window_size % 2 == 0:
			window_size += 1
		window_size = max(window_size, 3)

		thresh_sauvola = skimage.filters.threshold_sauvola(
			grayscale, window_size)
		binary = grayscale > thresh_sauvola

		dewarper = self._page.dewarper

		for prediction in self._segmentation.predictions:
			if prediction.type == PredictorType.SEPARATOR:
				bg = prediction.background_label.value
				mask = PIL.Image.fromarray(
					(prediction.labels != bg).astype(np.uint8) * 255)
				mask = dewarper.dewarp_image(mask, cv2.INTER_NEAREST)
				mask = skimage.morphology.binary_dilation(
					np.array(mask) > 0, skimage.morphology.square(3))
				binary = np.logical_or(binary, mask)

		#PIL.Image.fromarray(binary).save("/Users/arbeit/tmp.png")

		return binary.astype(np.float32)

	@cached_property
	def geometry(self):
		return self.page.geometry(dewarped=True)

	def union(self, shapes):
		return self._union(self._page, shapes)

	@property
	def unmodified_contours(self) -> dict:
		return self._unmodified_contours

	@property
	def contours(self) -> dict:
		return self._contours

	def _contour_name(self, contour):
		return self._names[id(contour)]

	def contour_path(self, contour):
		return tuple(self._contour_name(contour).split('/'))

	@property
	def warped_lines(self) -> dict:
		return self._warped_lines

	@cached_property
	def warped_lines_by_block(self) -> dict:
		lines_by_block = collections.defaultdict(list)
		for k, line in self._warped_lines.items():
			lines_by_block[k[:3]].append(line)
		return lines_by_block

	@property
	def by_predictors(self):
		by_predictors = collections.defaultdict(list)
		for k, contour in self._contours.items():
			by_predictors[k[:2]].append(contour)
		return by_predictors

	def line_count(self, a):
		return self._line_counts[a]

	def map(self, f):
		new_names = {}

		def named_f(k, c):
			contour = f(k, c)
			new_names[id(contour)] = "/".join(k)
			return contour

		self._contours = dict(
			(k, named_f(k, contour))
			for k, contour in self._contours.items())
		self._names = new_names

	def combine(self, sources, agg_path=None):
		contours = self._contours

		if agg_path is None:
			s = list(sources)
			i = int(np.argmax([contours[p].area for p in s]))
			agg_path = s[i]

		u = self.union([contours[p] for p in sources])
		self.modify_contour(agg_path, u)
		self._line_counts.combine(sources, agg_path)

		for k in sources:
			if k != agg_path:
				self.remove_contour(k)
				self._mapped_from[agg_path].append(k)

	def combine_from_graph(self, graph):
		if graph.number_of_edges() > 0:
			for nodes in nx.connected_components(graph):
				self.combine(nodes)
			return True
		else:
			return False

	def _set_contour(self, path, contour):
		old_contour = self._contours.get(path)
		if old_contour:
			del self._names[id(old_contour)]
		self._contours[path] = contour
		self._names[id(contour)] = "/".join(path)

	def modify_contour(self, path, contour):
		if contour.is_empty:
			self.remove_contour(path)
		else:
			self._set_contour(path, contour)

	def remove_contour(self, path):
		contour = self._contours[path]
		del self._names[id(contour)]
		del self._contours[path]
		self._line_counts.remove(path)

	def add_contour(self, label, contour):
		i = 1 + self._max_labels[label]
		self._max_labels[label] = i
		path = label + (str(i),)
		self._set_contour(path, contour)
		return path

	def sources(self, path):
		m = self._mapped_from.get(path)
		if m is None:
			return [path]
		else:
			sources = []
			for x in m:
				sources.extend(self.sources(x))
			return sources

	def line_heights(self, path):
		# note: we must not cache this function, since with
		# every combine() the result might change.

		dewarper = self.page.dewarper
		lines_by_block = self.warped_lines_by_block
		heights = []

		for source in self.sources(path):
			lines = lines_by_block.get(source, [])
			for line in lines:
				heights.append(line.dewarped_height(dewarper))

		return heights

	@cached_property
	def median_line_height(self):
		heights = []
		dewarper = self.page.dewarper
		for lines in self.warped_lines_by_block.values():
			for line in lines:
				heights.append(line.dewarped_height(dewarper))
		return max(6, int(np.median(heights)))


class Transformer:
	def __init__(self, operators):
		self._operators = operators

	def __call__(self, regions, callback=None):
		regions.check_geometries(allowed=["Polygon", "MultiPolygon"])

		for i, operator in enumerate(self._operators):
			try:
				operator(regions)
				regions.check_geometries(allowed=["Polygon"])
			except:
				logging.exception("error in %s in Transformer stage %d" % (
					operator.__class__.__name__, 1 + i))

			if callback:
				callback(i, regions)


def alignment(a0, a1, b0, b1, mode="min"):
	span_a = portion.closed(a0, a1)
	span_b = portion.closed(b0, b1)
	shared = span_a & span_b
	if shared.empty:
		return 0

	da = a1 - a0
	db = b1 - b0
	if mode == "min":
		d = min(da, db)
	elif mode == "a":
		d = da
	elif mode == "b":
		d = db
	else:
		raise ValueError(mode)

	return (shared.upper - shared.lower) / d


class IsOnSameLine:
	def __init__(
		self, max_line_count=3, cohesion=0.8,
		alignment=0.8, fringe=0, max_distance=0.006):

		self._max_line_count = max_line_count
		self._cohesion = cohesion
		self._min_alignment = alignment
		self._fringe = fringe
		self._max_distance = max_distance

	def for_regions(self, regions):
		return partial(self.check, regions=regions)

	def check(self, p, q, regions):
		lc = regions.line_count
		if max(lc(p), lc(q)) > self._max_line_count:
			return False

		contours = regions.contours
		a = contours[p]
		b = contours[q]

		_, ay0, _, ay1 = a.bounds
		_, by0, _, by1 = b.bounds

		if alignment(ay0, ay1, by0, by1) < self._min_alignment:
			return False

		if a.distance(b) > regions.geometry.rel_length(self._max_distance):
			return False

		u = regions.union([a, b])

		if regions.separators.check_obstacles(
			u.bounds, ["separators/V", "separators/T"], self._fringe):
			return False

		c = _cohesion([a, b], u)
		return c > self._cohesion


class IsBelow:
	def __init__(self, alignment=0.95):
		self._min_alignment = alignment

	def for_regions(self, regions):
		return partial(self.check, regions=regions)

	def _is_below(self, contour_a, contour_b, h):
		minxa, minya, maxxa, maxya = contour_a.bounds
		minxb, minyb, maxxb, maxyb = contour_b.bounds

		if not (0 < minyb - maxya < h):
			return False

		if alignment(minxa, maxxa, minxb, maxxb) < self._min_alignment:
			return False

		return True

	def check(self, path_a, path_b, regions):
		hs = regions.line_heights(path_a) + regions.line_heights(path_b)
		if len(hs) < 2:
			return False
		h = np.median(hs)

		contours = regions.contours
		a = contours[path_a]
		b = contours[path_b]

		return self._is_below(a, b, h) or self._is_below(b, a, h)


class Dilation:
	def __init__(self, spec):
		self._operator = HullOperator(spec)

	def __call__(self, regions):
		regions.map(lambda _, contour: self._operator(regions.page, contour))


class AdjacencyMerger:
	def __init__(self, filters, criterion):
		self._filter = RegionsFilter(filters)
		self._criterion = criterion

	def __call__(self, regions):
		should_merge = self._criterion.for_regions(regions)
		neighbors_ = neighbors(regions.contours)

		graph = nx.Graph()
		graph.add_nodes_from(regions.contours.keys())

		for p, q in neighbors_.edges():
			if self._filter(p) and self._filter(q):
				if should_merge(p, q):
					graph.add_edge(p, q)

		regions.combine_from_graph(graph)


class OverlapMerger:
	def __init__(self, maximum_overlap):
		self._maximum_overlap = maximum_overlap

	def _merge(self, regions, contours):
		graph = nx.Graph()
		graph.add_nodes_from([
			regions.contour_path(c) for c in contours])

		tree = shapely.strtree.STRtree(contours)
		for contour in contours:
			for other in tree.query(contour):
				if regions.contour_path(contour) == regions.contour_path(other):
					continue
				if overlap_ratio(contour, other) > self._maximum_overlap:
					graph.add_edge(
						regions.contour_path(contour),
						regions.contour_path(other))

		return regions.combine_from_graph(graph)

	def __call__(self, regions):
		modify = set(regions.by_predictors.keys())
		while modify:
			changed = set()
			for k, contours in regions.by_predictors.items():
				if k in modify:
					if self._merge(regions, contours):
						changed.add(k)
			modify = changed


class Overlap:
	def __init__(self, contours, active):
		polygons = []
		for path, polygon in contours.items():
			if path[:2] in active:
				polygons.append(polygon)
		self._tree = shapely.strtree.STRtree(polygons)

	def __call__(self, shape):
		t_areas = [0]
		for t in self._tree.query(shape):
			intersection = t.intersection(shape)
			if not intersection.is_empty:
				t_areas.append(intersection.area / t.area)
		return max(t_areas)


class DominanceOperator:
	def __init__(self, filters, fringe, strategy):
		self._filter = RegionsFilter(filters)
		self._fringe = fringe
		self._strategy = strategy

	def _graph(self, regions, contours):
		graph = nx.Graph()
		graph.add_nodes_from([
			regions.contour_path(c) for c in contours])

		tree = shapely.strtree.STRtree(contours)
		for contour in contours:
			for other in tree.query(contour):
				if regions.contour_path(contour) == regions.contour_path(other):
					continue
				if contour.intersects(other):
					graph.add_edge(
						regions.contour_path(contour),
						regions.contour_path(other))

		return graph

	def _resolve(self, regions, nodes):
		if len(nodes) <= 1:
			return False

		fringe = regions.geometry.rel_length(self._fringe)
		changed = False

		# phase 1. larger areas consume smaller areas.

		remaining = dict([
			(k, regions.contours[k].area)
			for k in nodes])

		def merge(union, agg_path):
			regions.combine(union, agg_path=agg_path)
			for x in union:
				if x != agg_path:
					del remaining[x]
			remaining[agg_path] = regions.contours[agg_path].area

		done = False
		while not done:
			by_area = [x[0] for x in sorted(
				list(remaining.items()), key=lambda x: x[-1])]

			done = True
			for i in reversed(range(1, len(by_area))):
				largest_path = by_area[i]
				largest = regions.contours[largest_path]
				largest = largest.buffer(fringe)
				union = [largest_path]
				for path in by_area[:i]:
					polygon = regions.contours[path]
					if polygon.is_empty:
						union.append(path)
					elif largest.contains(polygon):
						union.append(path)
				if len(union) > 1:
					merge(union, largest_path)
					done = False
					changed = True
					break

		# phase 2. resolve remaining overlaps.

		def modify(key, shape):
			if shape.geom_type == "Polygon":
				regions.modify_contour(key, shape)
				remaining[key] = shape.area
			elif shape.geom_type == "MultiPolygon":
				regions.remove_contour(key)
				del remaining[key]
				for geom in shape.geoms:
					new_path = regions.add_contour(key[:2], geom)
					remaining[new_path] = geom.area
			else:
				raise RuntimeError(
					"illegal shape geom_type %s" % shape.geom_type)

		def shrink(shrinked_path, constant_path):
			shape = regions.contours[shrinked_path]
			other = regions.contours[constant_path]

			intersection = shape.intersection(other)
			if intersection.area < 1:
				return False

			remaining_shape = shape.difference(other)
			if remaining_shape.is_empty:
				regions.remove_contour(shrinked_path)
				del remaining[shrinked_path]
			else:
				modify(shrinked_path, remaining_shape)

			return True

		done = len(remaining) < 2
		while not done:
			neighbors_ = neighbors(dict(
				(k, regions.contours[k]) for k in remaining.keys()))

			done = True
			for pk, qk in neighbors_.edges():
				if pk not in regions.contours or qk not in regions.contours:
					continue

				intersection = regions.contours[pk].intersection(regions.contours[qk])
				if intersection.area < 1:
					continue

				done = False
				changed = True

				r = self._strategy(regions.contours, pk, qk)

				if r[0] == "merge":
					merge([pk, qk], r[1])
				elif r[0] == "split":
					shrink(r[1], r[2])
				elif r[0] == "custom":
					ps, qs = r[1]
					modify(pk, ps)
					modify(qk, qs)
				else:
					raise ValueError(r)

		return changed

	def __call__(self, regions):
		f_contours = [
			v for k, v in regions.contours.items()
			if self._filter(k)]

		graph = self._graph(regions, f_contours)

		for nodes in nx.connected_components(graph):
			self._resolve(regions, nodes)


class HullOperator:
	def __init__(self, spec):
		names = ("none", "rect", "convex", "concave")

		funcs = dict(
			(x, getattr(HullOperator, "_" + x))
			for x in names)

		self._f = build_func_from_string(spec, funcs)

	@staticmethod
	def _none(page, shape):
		if shape.geom_type != "Polygon":
			return shape.convex_hull
		else:
			return shape

	@staticmethod
	def _rect(page, shape):
		return shapely.geometry.box(*shape.bounds)

	@staticmethod
	def _convex(page, shape):
		return shape.convex_hull

	@staticmethod
	def _concave(page, shape, concavity=2, detail=0.01):
		from origami.concaveman import concaveman2d

		if shape.geom_type == "MultiPolygon":
			coords = []
			for geom in shape.geoms:
				coords.extend(np.asarray(geom.exterior))
		elif shape.geom_type == "Polygon":
			coords = np.asarray(shape.exterior)
		else:
			raise RuntimeError(
				"unexpected geom_type %s" % shape.geom_type)

		detail = page.geometry(dewarped=True).rel_length(detail)
		pts = concaveman2d(
			coords,
			scipy.spatial.ConvexHull(coords).vertices,
			concavity=concavity,
			lengthThreshold=detail)

		shape1 = shapely.geometry.Polygon(pts)
		shape1 = shape1.union(shape)
		return shape1

	def __call__(self, page, shape):
		return self._f(page, shape)


class UnionOperator:
	def __init__(self, spec):
		self._dilation = HullOperator(spec)

	def __call__(self, page, shapes):
		if len(shapes) > 1:
			u = shapely.ops.unary_union(shapes)
		else:
			u = shapes[0]

		return self._dilation(page, u)


class SetUnionOperator:
	def __init__(self, spec):
		self._union = UnionOperator(spec)

	def __call__(self, regions):
		regions.set_union_operator(self._union)


class SequentialMerger:
	def __init__(self, filters, cohesion, max_distance, max_error, fringe, obstacles):
		self._filter = RegionsFilter(filters)
		self._cohesion = cohesion
		self._max_distance = max_distance
		self._max_error = max_error
		self._fringe = fringe
		self._obstacles = obstacles

	def _merge(self, regions, names, error_overlap):
		contours = regions.contours
		shapes = [contours[x] for x in names]

		fringe = regions.geometry.rel_length(self._fringe)
		label = names[0][:2]
		assert(all(x[:2] == label for x in names))

		graph = nx.Graph()
		graph.add_nodes_from(names)

		max_distance = regions.geometry.rel_length(
			self._max_distance)

		def union(i, j):
			return regions.union(shapes[i:j])

		i = 0
		while i < len(shapes):
			good = False
			for j in range(i + 1, len(shapes)):
				d = union(i, j).distance(shapes[j])

				if d > max_distance:
					break

				u = union(i, j + 1)

				if regions.separators.check_obstacles(
					u.bounds, self._obstacles, fringe):
					break

				cohesion = _cohesion(shapes[i:j + 1], u)
				error = error_overlap(u)

				if cohesion < self._cohesion[0] or error > self._max_error:
					break
				elif cohesion > self._cohesion[1]:
					for k in range(i, j):
						graph.add_edge(names[k], names[k + 1])
					shapes[j] = u
					i = j
					good = True
					break

			if not good:
				i += 1

		return regions.combine_from_graph(graph)

	def _compute_order(self, regions, contours):
		fringe = regions.geometry.rel_length(self._fringe)
		order = polygon_order(list(regions.contours.items()), fringe=fringe)
		selection = set(regions.contour_path(c) for c in contours)
		return [x for x in order if x in selection]

	def _merge_pass(self, regions, by_predictors):
		merged = set()

		for path, contours in by_predictors.items():
			if not self._filter(path):
				continue

			order = self._compute_order(
				regions, contours)

			labels = set(by_predictors.keys())
			error_overlap = Overlap(
				regions.unmodified_contours,
				labels - set([path[:2]]))

			if self._merge(
				regions,
				order,
				error_overlap):
				merged.add(path)

		return merged

	def __call__(self, regions):
		by_predictors = regions.by_predictors

		while by_predictors:
			dirty = self._merge_pass(
				regions, by_predictors)

			if not dirty:
				break

			by_predictors = regions.by_predictors
			keep = set(by_predictors.keys()) & dirty
			by_predictors = dict(
				(k, v)
				for k, v in by_predictors.items()
				if k in keep)


class Shrinker:
	def __init__(self, min_area=0):
		self._min_area = min_area

	def __call__(self, regions):
		by_labels_nomod = collections.defaultdict(list)
		for k, contour in regions.unmodified_contours.items():
			by_labels_nomod[k[:2]].append(contour)

		min_area = regions.geometry.rel_area(self._min_area)
		for k0, v0 in by_labels_nomod.items():
			tree = shapely.strtree.STRtree(v0)
			for k, contour in list(regions.contours.items()):
				if k[:2] != k0[:2]:
					continue
				try:
					q = tree.query(contour)
					if q:
						bounds = shapely.ops.unary_union(q).bounds
						box = shapely.geometry.box(*bounds)
						modified = box.intersection(contour)
						if modified.area >= min_area:
							regions.modify_contour(k, modified)
						else:
							regions.remove_contour(k)
				except ValueError:
					logging.exception("deformed geometry errors")


class AreaFilter:
	def __init__(self, min_area):
		self._min_area = min_area

	def __call__(self, regions):
		min_area = regions.geometry.rel_area(self._min_area)
		remove = []
		for k, contour in regions.contours.items():
			if contour.area < min_area:
				remove.append(k)
		for k in remove:
			regions.remove_contour(k)


def crop(pixels, contour):
	minx, miny, maxx, maxy = contour.bounds

	miny = int(max(0, miny))
	minx = int(max(0, minx))
	maxy = int(min(maxy, pixels.shape[0]))
	maxx = int(min(maxx, pixels.shape[1]))

	return pixels[miny:maxy, minx:maxx], (minx, miny)


class FixSpillOver:
	def _crop(self, regions, contour):
		return crop(regions.grayscale, contour)

	def _binarized_crop(self, regions, contour):
		return crop(regions.binarized, contour)


class SplitFilter:
	def __init__(self, min_area=0.2):
		self._min_area = min_area

	def __call__(self, union, shapes):
		union_area = union.area
		min_area = min([shape.area for shape in shapes])
		return min_area >= union_area * self._min_area


class SplitDetector:
	def __init__(self, quantile=0.9, smooth=1, intensity=0.05, width=2, border=0.1):
		self._quantile = quantile
		self._smooth = smooth
		self._intensity = intensity
		self._width = width
		self._border = border

	def __call__(self, pixels, scale):
		if pixels.dtype == np.uint8:
			pixels = pixels.astype(np.float32) / 255.0

		assert pixels.dtype == np.float32

		freq, dens = scipy.signal.periodogram(
			pixels, axis=0)

		ink_h = scipy.ndimage.convolve(
			np.quantile(dens, self._quantile, axis=0),
			kernel(int(self._smooth * scale)),
			mode="nearest")

		span = int(self._border * len(ink_h))
		ink_h[:span] = 0
		ink_h[-span:] = 0

		peaks, info = scipy.signal.find_peaks(
			-ink_h, height=-self._intensity, distance=int(self._width * scale))

		return peaks, info


class FixSpillOverH(FixSpillOver):
	def __init__(
			self, filters, split_detector=SplitDetector(),
			min_line_count=3, split_filter=SplitFilter()):

		self._filter = RegionsFilter(filters)
		self._split_detector = split_detector
		self._min_line_count = min_line_count
		self._split_filter = split_filter

	def __call__(self, regions):
		# for each contour, sample the binarized image.
		# if we detect a whitespace region in a column,
		# we split the polygon. since we dewarped, we know
		# columns are unskewed.

		splits = []
		binarized = regions.binarized

		for k, contour in regions.contours.items():
			if not self._filter(k):
				continue

			if regions.line_count(k) < self._min_line_count:
				continue

			line_heights = regions.line_heights(k)
			if not line_heights:
				continue

			line_height = np.median(line_heights)

			crop, (minx, miny) = self._crop(regions, contour)
			peaks, info = self._split_detector(crop, scale=line_height)

			if len(peaks) > 0:
				i = np.argmax(info["peak_heights"])
				x = peaks[i] + minx

				sep = shapely.geometry.LineString([
					[x, -1], [x, binarized.shape[0] + 1]
				])

				splits.append((k, contour, sep, line_height))

		for k, contour, sep, lh in splits:
			split_length = _line_length(contour.intersection(sep))
			if split_length < lh * self._min_line_count:
				continue

			shapes = shapely.ops.split(contour, sep).geoms
			if self._split_filter(contour, shapes):
				regions.remove_contour(k)
				for shape in shapes:
					regions.add_contour(k[:2], shape)


class FixSpillOverHOnSeparator(FixSpillOver):
	def __init__(self, detector, split_filter=SplitFilter()):
		self._detector = detector  # RegionSeparatorDetector
		self._split_filter = split_filter

	def __call__(self, regions):
		page_w, page_h = regions.geometry.size
		dividers = self._detector(regions)

		for k, xs in dividers.items():
			if not xs:
				continue

			remaining = regions.contours[k]
			split_shapes = []

			for x in xs:
				sep = shapely.geometry.LineString([
					[x, -1], [x, page_h + 1]
				])

				shapes = shapely.ops.split(remaining, sep).geoms

				if self._split_filter(remaining, shapes):
					polygons = []
					ok = True
					for shape in shapes:
						if shape.geom_type == "Polygon":
							polygons.append(shape)
						else:
							ok = False
							break
					if ok:
						polygons = sorted(polygons, key=lambda p: p.bounds[0])
						split_shapes.extend(polygons[:-1])
						remaining = polygons[-1]

			if split_shapes:
				regions.remove_contour(k)

				for shape in split_shapes:
					regions.add_contour(k[:2], shape)

				regions.add_contour(k[:2], remaining)


class FixSpillOverV(FixSpillOver):
	def __init__(self, filters, split_detector=SplitDetector()):
		self._filter = RegionsFilter(filters)
		self._split_detector = split_detector

	def __call__(self, regions):
		median_lh = regions.median_line_height

		splits = []
		binarized = regions.binarized

		for k, contour in regions.contours.items():
			if not self._filter(k):
				continue

			crop, (minx, miny) = self._crop(regions, contour)
			peaks, info = self._split_detector(
				crop.transpose(), scale=median_lh)

			if len(peaks) > 0:
				i = np.argmax(info["peak_heights"])
				y = peaks[i] + miny

				sep = shapely.geometry.LineString([
					[-1, y], [binarized.shape[1] + 1, y]
				])

				splits.append((k, contour, sep))

		for k, contour, sep in splits:
			shapes = shapely.ops.split(contour, sep).geoms
			regions.remove_contour(k)
			for shape in shapes:
				regions.add_contour(k[:2], shape)


def shapely_limits(geom, axis):
	bbox = np.array(geom.bounds)
	return bbox.reshape((2, 2)).T[axis]


class RegionSeparatorDetector:
	def __init__(self, filters, label, axis, min_distance=20, coverage_ratio=0.3):
		self._filter = RegionsFilter(filters)
		self._label = label
		self._axis = axis
		self._min_distance = min_distance
		self._coverage_ratio = coverage_ratio

	def __call__(self, regions):
		contours = dict(
			(k, v) for k, v
			in regions.contours.items()
			if self._filter(k))

		for k, contour in contours.items():
			assert regions.contour_path(contour) == k

		tree = shapely.strtree.STRtree(
			list(contours.values()))
		seps = collections.defaultdict(list)

		# as dewarping has happened before, we assume separator
		# lines are basically straight horizontals or verticals,
		# with only very small variations. this simplifies the
		# following algorithm a lot.

		for sep in regions.separators.for_label(self._label):
			for contour in tree.query(sep):
				sep_i = contour.intersection(sep)
				if sep_i and sep_i.geom_type == "LineString":
					path = regions.contour_path(contour)
					coords = np.array(sep_i.coords)
					mx = np.median(coords[:, self._axis])
					miny = np.min(coords[:, 1 - self._axis])
					maxy = np.max(coords[:, 1 - self._axis])
					seps[path].append((mx, miny, maxy))

		agg = sklearn.cluster.AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=self._min_distance,
			affinity="l1",
			linkage="average",
			compute_full_tree=True)

		columns = dict()

		for path, entries in seps.items():
			entries = np.array(entries)

			if entries.shape[0] > 1:
				clustering = agg.fit([(x, 0) for x in entries[:, 0]])
				labels = clustering.labels_
			else:
				labels = np.array([0])

			cx = []

			for i in range(np.max(labels) + 1):
				sep_x = np.median(entries[labels == i, 0])

				# investigate actual coverage at sep_x.
				coverage = intervaltree.IntervalTree()
				for miny, maxy in entries[labels == i, 1:]:
					coverage.addi(miny, maxy + 1, True)
				coverage.merge_overlaps(strict=False)

				cmin, cmax = shapely_limits(contours[path], 1 - self._axis)
				coords = np.zeros((2, 2), dtype=np.float64)
				coords[:, self._axis] = sep_x
				coords[:, 1 - self._axis] = (cmin - 1, cmax + 1)
				divider = shapely.geometry.LineString(coords)
				divider = contours[path].intersection(divider)

				if divider and divider.geom_type == "LineString":
					dmin, dmax = shapely_limits(divider, 1 - self._axis)
					dlen = dmax - dmin

					clen = 0
					for iv in coverage:
						lo = max(iv.begin, dmin)
						hi = min(iv.end, dmax)
						clen += max(0, hi - lo)

					coverage_ratio = clen / dlen
					if coverage_ratio > self._coverage_ratio:
						cx.append(sep_x)

			columns[path] = sorted(cx)

		return columns


def divide(shape, dividers, axis):
	if not dividers:
		return [shape]

	rest = shape
	areas = []
	for divider in sorted(dividers):
		bounds = np.array(shape.bounds).reshape((2, 2))

		p0 = bounds[0] - np.array([1, 1])
		p1 = bounds[1] + np.array([1, 1])
		p0[axis] = divider
		p1[axis] = divider

		line = shapely.geometry.LineString([p0, p1])
		if not rest.is_empty:
			items = shapely.ops.split(rest, line)
		else:
			items = []

		bins = [[], []]
		for geom in items:
			coords = list(geom.centroid.coords)[0]
			is_before = coords[axis] - divider < 0
			bins[0 if is_before else 1].append(geom)

		parts = []
		for i in (0, 1):
			geoms = bins[i]
			if len(geoms) > 1:
				part = shapely.ops.unary_union(geoms).convex_hull
			elif len(geoms) == 1:
				part = geoms[0]
			else:
				part = shapely.geometry.GeometryCollection()  # empty
			parts.append(part)

		areas.append(parts[0])
		rest = parts[1]

	areas.append(rest)
	return areas


def find_table_headers(areas, line_h):
	if line_h is None:
		return
	for i, area in enumerate(areas):
		if area.geom_type == "Polygon":
			_, miny, _, maxy = area.bounds
			if maxy - miny < 3 * line_h:
				yield i


def map_dict(values, mapping):
	mapped_values = dict()
	for k, v in values.items():
		for k2 in mapping.get(k, [k]):
			mapped_values[k2] = v
	return mapped_values


def subdivide_table_blocks(filters, regions, columns, dividers):
	split_map = collections.defaultdict(list)
	split_contours = dict()

	contours = regions.contours
	filter_ = RegionsFilter(filters)

	for k, contour in contours.items():
		if not filter_(k):
			split_contours[k] = contour
			continue

		block_path = k[:3]
		block_id = block_path[-1]

		def make_id(division, row, column):
			pos = (division, row, column)
			pos = list(map(str, filter(lambda x: x, pos)))
			return "%s.%s" % (block_id, ".".join(pos))

		line_hs = regions.line_heights(k)
		line_h = np.median(line_hs) if len(line_hs) >= 2 else None

		# this whole logic is intertwined with LineExtractor. for
		# each table division (collection of rows), we produce one
		# block. for this block, we will detect baselines (for the
		# whole width and across columns) in the LINES stages. each
		# baseline will then we split along the available columns in
		# LineExtractor to generate the content of each cell.

		# for line headers, we follow a different approach, since they
		# might have mixed one- and multi-line content. we explicitly
		# split them here into separate blocks for each column.

		# as a result, a three column table with header would be split
		# into four blocks (with X being the region id of the table and
		# with X.2.1.1 getting subdivided later inside LineExtractor):

		# ----------------------------------------------------------
		# | X.1.1.1  | X.1.1.2 | X.1.1.3                           |
		# ----------------------------------------------------------
		# | X.2.1.1                                                |
		# |                                                        |
		# |                                                        |
		# |                                                        |
		# ----------------------------------------------------------

		areas = divide(contour, dividers.get(k, []), 1)
		for i in list(find_table_headers(areas, line_h)):
			areas[i] = divide(areas[i], columns.get(k, []), 0)

		def split_block(split_block_id, area, add_to_map):
			split_k = block_path[:2] + (split_block_id,)
			if add_to_map:
				split_map[k].append(split_k)
			split_contours[split_k] = area

		for i, area_y in enumerate(areas):
			if isinstance(area_y, list):
				for j, area_xy in enumerate(area_y):
					split_block(make_id(
						i + 1, 1, j + 1), area_xy, False)
			elif k in columns:
				# id will get rewritten for various columns
				# inside LineExtractor._column_path
				split_block(make_id(i + 1, 1, 1), area_y, True)
			else:
				# happens if we have a table without any
				# detected T separators.
				split_block(make_id(i + 1, 1, 1), area_y, False)

	return split_contours, map_dict(columns, split_map), map_dict(dividers, split_map)


def to_table_data_dict(items):
	return dict(
		("/".join(path), [round(x, 1) for x in xs])
		for path, xs in items.items())


class LayoutDetectionProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

		layout_name = options["layout"]

		try:
			imported_module = importlib.import_module("origami.custom.layouts.%s" % layout_name)
		except ModuleNotFoundError as e:
			raise click.UsageError("layout %s not found in origami.custom.layouts" % layout_name)

		self._transformer = getattr(imported_module, "make_transformer")()

		self._table_column_detector = RegionSeparatorDetector(
			"regions/TABULAR", "separators/T", axis=0)
		self._table_divider_detector = RegionSeparatorDetector(
			"regions/TABULAR", "separators/H", axis=1)

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(
				Artifact.CONTOURS, Artifact.LINES, Artifact.SEGMENTATION,
				stage=Stage.WARPED)),
			("dewarped", Input(
				Artifact.CONTOURS,
				stage=Stage.DEWARPED)),
			("output", Output(
				Artifact.CONTOURS, Artifact.TABLES,
				stage=Stage.AGGREGATE))
		]

	def process(self, page_path: Path, warped, dewarped, output):
		blocks = dewarped.regions.by_path

		if not blocks:
			return

		separators = dewarped.separators

		page = dewarped.page
		contours = [(k, block.image_space_polygon) for k, block in blocks.items()]

		def save_transformer_stage(i, current_regions):
			current_regions.debug_save(
				output.data_path / f"layout_after_step_{i + 1}.json")

		regions = Regions(
			page, warped.lines.by_path,
			contours, separators,
			warped.segmentation)
		self._transformer(regions)  #, callback=save_transformer_stage)

		# we split table cells into separate regions so that the
		# next stage (baseline detection) runs on isolated divisions.
		# see subdivide_table_blocks for details.

		split_contours, columns, dividers = subdivide_table_blocks(
			"regions/TABULAR",
			regions,
			columns=self._table_column_detector(regions),
			dividers=self._table_divider_detector(regions))

		output.tables(dict(
			version=1,
			columns=to_table_data_dict(columns),
			dividers=to_table_data_dict(dividers)))

		with output.contours(copy_meta_from=dewarped) as zf:
			for path, shape in split_contours.items():
				if shape.geom_type != "Polygon" and not shape.is_empty:
					logging.info("contour %s is %s" % (path, shape.geom_type))
				zf.writestr("/".join(path) + ".wkt", shape.wkt.encode("utf8"))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--layout',
	type=str,
	default="bbz",
	help="Name of set of heuristic layout rules to apply.")
@Processor.options
def detect_layout(data_path, **kwargs):
	""" Detect layout and reading order for documents in DATA_PATH. """
	processor = LayoutDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_layout()

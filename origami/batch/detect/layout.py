#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.spatial
import click
import shapely.ops
import shapely.wkt
import shapely.strtree
import shapely.geometry
import shapely.ops
import sklearn.cluster
import skimage.filters
import networkx as nx
import collections
import portion
import logging
import importlib

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
	def __init__(self, page, warped_lines, contours, separators, union):
		self._page = page

		self._contours = dict(non_empty_contours(contours))
		self._unmodified_contours = self._contours.copy()
		for k, contour in contours:
			contour.name = "/".join(k)
		self.separators = separators

		self._line_counts = LineCounts(warped_lines)
		self._warped_lines = warped_lines
		self._union = union
		self._mapped_from = collections.defaultdict(list)

		max_labels = collections.defaultdict(int)
		for k in self._contours.keys():
			max_labels[k[:2]] = max(max_labels[k[:2]], int(k[2]))
		self._max_labels = max_labels

	def check_geometries(self, allowed):
		for k, contour in self._contours.items():
			if not contour.is_valid:
				raise ValueError("invalid contour")
			if contour.geom_type not in allowed:
				raise ValueError("%s not in %s" % (
					contour.geom_type, allowed))

	@property
	def page(self):
		return self._page

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
		def named_f(k, c):
			contour = f(k, c)
			contour.name = "/".join(k)
			return contour

		self._contours = dict(
			(k, named_f(k, contour))
			for k, contour in self._contours.items())

	def combine(self, sources, agg_path=None):
		contours = self._contours

		if agg_path is None:
			s = list(sources)
			i = np.argmax([contours[p].area for p in s])
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

	def modify_contour(self, path, contour):
		if contour.is_empty:
			self.remove_contour(path)
		else:
			self._contours[path] = contour
			contour.name = "/".join(path)

	def remove_contour(self, path):
		del self._contours[path]
		self._line_counts.remove(path)

	def add_contour(self, label, contour):
		i = 1 + self._max_labels[label]
		self._max_labels[label] = i
		path = label + (str(i),)
		self._contours[path] = contour
		contour.name = "/".join(path)
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


class Transformer:
	def __init__(self, operators):
		self._operators = operators

	def __call__(self, regions):
		regions.check_geometries(allowed=["Polygon", "MultiPolygon"])

		for i, operator in enumerate(self._operators):
			try:
				operator(regions)
				regions.check_geometries(allowed=["Polygon"])
			except:
				logging.exception("error in %s in Transformer stage %d" % (
					operator.__class__.__name__, 1 + i))


def _alignment(a0, a1, b0, b1):
	span_a = portion.closed(a0, a1)
	span_b = portion.closed(b0, b1)
	shared = span_a & span_b
	if shared.empty:
		return 0

	return (shared.upper - shared.lower) / min(
		a1 - a0, b1 - b0)


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

		if _alignment(ay0, ay1, by0, by1) < self._min_alignment:
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

		if _alignment(minxa, maxxa, minxb, maxxb) < self._min_alignment:
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
		self._operator = DilationOperator(spec)

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
			tuple(c.name.split("/")) for c in contours])

		tree = shapely.strtree.STRtree(contours)
		for contour in contours:
			for other in tree.query(contour):
				if contour.name == other.name:
					continue
				if overlap_ratio(contour, other) > self._maximum_overlap:
					graph.add_edge(
						tuple(contour.name.split("/")),
						tuple(other.name.split("/")))

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
	def __init__(self, filters, fringe):
		self._filter = RegionsFilter(filters)
		self._fringe = fringe

	def _graph(self, contours):
		graph = nx.Graph()
		graph.add_nodes_from([
			tuple(c.name.split("/")) for c in contours])

		tree = shapely.strtree.STRtree(contours)
		for contour in contours:
			for other in tree.query(contour):
				if contour.name == other.name:
					continue
				graph.add_edge(
					tuple(contour.name.split("/")),
					tuple(other.name.split("/")))

		return graph

	def _resolve(self, regions, nodes):
		fringe = regions.geometry.rel_length(self._fringe)

		# phase 1. larger areas consume smaller areas.

		remaining = dict([
			(k, regions.contours[k].area)
			for k in nodes])

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
				for path in by_area[:i - 1]:
					polygon = regions.contours[path]
					if polygon.is_empty:
						union.append(path)
					elif largest.contains(polygon):
						union.append(path)
				if len(union) > 1:
					regions.combine(union, agg_path=largest_path)
					for x in union:
						if x != largest_path:
							del remaining[x]
					remaining[largest_path] = regions.contours[largest_path].area
					done = False
					break

		# phase 2. resolve remaining overlaps.

		done = len(remaining) < 2
		while not done:
			by_area = [x[0] for x in sorted(
				list(remaining.items()), key=lambda x: x[-1])]

			done = True
			largest_path = by_area[-1]
			largest = regions.contours[largest_path]
			for path in by_area[:-1]:
				polygon = regions.contours[path]
				intersection = polygon.intersection(largest)
				if intersection.area > 1:
					done = False
					polygon = polygon.difference(largest)
					if polygon.is_empty:
						regions.remove_contour(path)
						del remaining[path]
					elif polygon.geom_type == "MultiPolygon":
						regions.remove_contour(path)
						del remaining[path]
						for geom in polygon.geoms:
							new_path = regions.add_contour(path[:2], geom)
							remaining[new_path] = geom.area
					elif polygon.geom_type != "Polygon":
						raise RuntimeError(
							"illegal shape %d" % polygon.geom_type)
					else:
						regions.modify_contour(path, polygon)
						remaining[path] = polygon.area

	def __call__(self, regions):
		f_contours = [
			v for k, v in regions.contours.items()
			if self._filter(k)]

		graph = self._graph(f_contours)

		for nodes in nx.connected_components(graph):
			self._resolve(regions, nodes)


class DilationOperator:
	def __init__(self, spec):
		names = ("none", "rect", "convex", "concave")

		funcs = dict(
			(x, getattr(DilationOperator, "_" + x))
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
		self._dilation = DilationOperator(spec)

	def __call__(self, page, shapes):
		if len(shapes) > 1:
			u = shapely.ops.cascaded_union(shapes)
		else:
			u = shapes[0]

		return self._dilation(page, u)


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
		selection = set(tuple(c.name.split("/")) for c in contours)
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
						bounds = shapely.ops.cascaded_union(q).bounds
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


class FixSpillOver:
	def __init__(
			self, filters,
			level=0.1, band=0.01, peak=0.8, whratio=1.5,
			min_line_count=3, min_area=0.2, window_size=15):

		# "whratio" is measured in line height (lh). examples:
		# good split: w=90, lh=35, whratio=2.5
		# bad split: w=30, lh=40, whratio=0.75

		self._filter = RegionsFilter(filters)
		self._level = level
		self._band = band
		self._peak = peak
		self._whratio = whratio
		self._min_line_count = min_line_count
		self._min_area = min_area
		self._window_size = window_size

	def _good_split(self, union, shapes):
		union_area = union.area
		min_area = min([shape.area for shape in shapes])
		return min_area >= union_area * self._min_area

	def __call__(self, regions):
		# for each contour, sample the binarized image.
		# if we detect a whitespace region in a column,
		# we split the polygon here and now.
		# since we dewarped, we know columns are unskewed.
		page = regions.page
		pixels = np.array(page.dewarped.convert("L"))

		kernel_w = max(
			10,  # pixels in warped image space
			int(np.ceil(regions.geometry.rel_length(self._band))))

		splits = []

		for k, contour in regions.contours.items():
			if not self._filter(k):
				continue

			if regions.line_count(k) < self._min_line_count:
				continue

			line_heights = regions.line_heights(k)
			if not line_heights:
				continue

			line_height = np.median(line_heights)
			int_line_height = max(6, int(line_height))

			minx, miny, maxx, maxy = contour.bounds

			miny = int(max(0, miny))
			minx = int(max(0, minx))
			maxy = int(min(maxy, pixels.shape[0]))
			maxx = int(min(maxx, pixels.shape[1]))

			crop = pixels[miny:maxy, minx:maxx]

			thresh_sauvola = skimage.filters.threshold_sauvola(
				crop, window_size=self._window_size)
			crop = (crop > thresh_sauvola).astype(np.float32)

			ink_v = scipy.ndimage.convolve(
				crop, kernel(int_line_height, 1), mode="constant", cval=1)

			ink_h = np.quantile(ink_v, self._level, axis=0)

			ink_h = np.convolve(
				ink_h, kernel(kernel_w), mode="same")

			peaks, info = scipy.signal.find_peaks(
				ink_h,
				height=self._peak,
				width=line_height * self._whratio,
				rel_height=1)

			if len(peaks) > 0:
				i = np.argmax(info["peak_heights"])
				x = peaks[i] + minx

				sep = shapely.geometry.LineString([
					[x, -1], [x, pixels.shape[0] + 1]
				])

				splits.append((k, contour, sep))

		for k, contour, sep in splits:
			shapes = shapely.ops.split(contour, sep).geoms
			if self._good_split(contour, shapes):
				regions.remove_contour(k)
				for shape in shapes:
					regions.add_contour(k[:2], shape)


class TableLayoutDetector:
	def __init__(self, filters, label, axis, min_distance=20):
		self._filter = RegionsFilter(filters)
		self._label = label
		self._axis = axis
		self._min_distance = min_distance

	def __call__(self, regions):
		contours = dict(
			(k, v) for k, v
			in regions.contours.items()
			if self._filter(k))

		for k, contour in contours.items():
			assert contour.name == "/".join(k)

		tree = shapely.strtree.STRtree(
			list(contours.values()))
		seps = collections.defaultdict(list)

		for sep in regions.separators.for_label(self._label):
			for contour in tree.query(sep):
				if contour.intersects(sep):
					path = tuple(contour.name.split("/"))
					mx = np.median(np.array(sep.coords)[:, self._axis])
					seps[path].append(mx)

		agg = sklearn.cluster.AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=self._min_distance,
			affinity="l1",
			linkage="average",
			compute_full_tree=True)

		columns = dict()

		for path, xs in seps.items():
			if len(xs) > 1:
				clustering = agg.fit([(x, 0) for x in xs])
				labels = clustering.labels_
				xs = np.array(xs)
				cx = []
				for i in range(np.max(labels) + 1):
					cx.append(np.median(xs[labels == i]))
				columns[path] = sorted(cx)
			else:
				columns[path] = [xs[0]]

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
				part = shapely.ops.cascaded_union(geoms).convex_hull
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

		self._union = getattr(imported_module, "make_union_operator")()
		self._transformer = getattr(imported_module, "make_transformer")()

		self._table_column_detector = TableLayoutDetector(
			"regions/TABULAR", "separators/T", axis=0)
		self._table_divider_detector = TableLayoutDetector(
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

		separators = Separators(
			warped.segmentation, dewarped.separators)

		page = dewarped.page
		contours = [(k, block.image_space_polygon) for k, block in blocks.items()]

		regions = Regions(
			page, warped.lines.by_path,
			contours, separators,
			self._union)
		self._transformer(regions)

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

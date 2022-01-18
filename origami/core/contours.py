import cv2
import numpy as np
import networkx as nx
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.affinity
import functools
import types
import itertools
import logging
import collections
import scipy
import skimage
import skimage.morphology
import skimage.measure
import semantic_version
import scipy.optimize

from heapq import heappush, heappop

from origami.core.neighbors import neighbors

from origami.core.polyline import FastPolylineFactory
from origami.core.polyline.skgeom import extract_simple_polygons, SkGeomMultiPolylineFactory


def _without_closing_point(pts):
	if tuple(pts[0]) == tuple(pts[-1]):
		pts = pts[:-1]

	return pts


def blowup(shape, area):
	def f(x):
		return abs(shape.buffer(x).area - area)

	opt = scipy.optimize.minimize_scalar(
		f, (0, 1), tol=0.01, options=dict(maxiter=100))

	if opt.success:
		return shape.buffer(opt.x)
	else:
		return shape


def fix_contour_pts(pts):
	pts = pts.reshape((len(pts), 2))

	pts = _without_closing_point(pts)

	if len(pts) < 3:
		return None

	return pts


def find_contours(mask):
	result = cv2.findContours(
		mask.astype(np.uint8),
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	v = semantic_version.Version(cv2.__version__)
	if v.major == 3:
		image, contours, hierarchy = result
	else:
		contours, hierarchy = result

	contours = [fix_contour_pts(pts) for pts in contours]

	return [c for c in contours if c is not None]


def selective_glue(polygons, glue_area):
	blobs = []
	regions = []

	blobs_q = set()
	small_blobs = []

	for i, polygon in enumerate(polygons):
		if polygon.area < glue_area:
			blob = blowup(polygon, glue_area)
			blob.name = str(i)
			blobs.append(blob)
			blobs_q.add(i)
			small_blobs.append(polygon)
		else:
			polygon.name = str(i)
			regions.append(polygon)

	graph = nx.Graph()
	graph.add_nodes_from(list(range(len(polygons))))

	tree = shapely.strtree.STRtree(regions + blobs)

	for blob in blobs:
		for region in tree.query(blob):
			if blob.name != region.name and region.intersects(blob):
				graph.add_edge(int(blob.name), int(region.name))

	results = []
	for names in nx.connected_components(graph):
		names = set(names) - blobs_q
		if len(names) == 1:
			results.append(polygons[list(names)[0]])
		elif len(names) > 1:
			results.append(shapely.ops.unary_union([
				polygons[i] for i in names
			]).convex_hull)

	tree = shapely.strtree.STRtree(results)
	for blob in small_blobs:
		if not any(x.contains(blob) for x in tree.query(blob)):
			results.append(blob)

	return results


class Contours:
	def __init__(self, ink=None, glue=0, buffer=0):
		# "ink" allows the caller to define areas that are considered
		# not connected, partially overriding the mask provided later.
		# ink shall contain connected components as they should be.
		self._ink = ink

		self._glue = glue
		self._buffer = buffer

	def __call__(self, mask):
		if self._ink is not None:
			ink = cv2.resize(
				self._ink.astype(np.uint8),
				tuple(reversed(mask.shape)),
				interpolation=cv2.INTER_NEAREST) > 0

			mask = np.logical_and(mask, ink)

		polygons = []
		for pts in find_contours(mask):
			polygons.append(shapely.geometry.Polygon(pts))

		if self._glue > 0:
			glue_area = mask.size * (self._glue ** 2)
			polygons = selective_glue(polygons, glue_area)

		for polygon in polygons:
			if self._buffer > 0:
				polygon = polygon.buffer(self._buffer)
				if polygon.geom_type != "Polygon":
					polygon = polygon.convex_hull
			yield polygon


class Decompose:
	def __call__(self, polygon):
		if not polygon.is_valid:
			for q in extract_simple_polygons(polygon.exterior.coords):
				assert q.is_valid
				yield q
		else:
			yield polygon


class Simplify:
	def __init__(self, tolerance):
		self._tolerance = tolerance

	def __call__(self, polygon):
		p = polygon.simplify(self._tolerance)
		if p and not p.is_empty:
			yield p


class FilterByArea:
	def __init__(self, min_area):
		self._min_area = min_area

	def __call__(self, polygon):
		if polygon.area >= self._min_area:
			yield polygon
		else:
			yield None


class WhiteSpaceProfiler:
	def __init__(self, page, black_threshold=0.4):
		self._black_threshold = black_threshold

		self._page = page
		self._binarized = self._page.binarized

		l_size = np.array(self._page.layout.blk.pixels.shape, dtype=np.float64)
		p_size = np.array(self._binarized.shape, dtype=np.float64)
		self._layout_to_page_scale = p_size / l_size

	def extract(self, p, size=100):
		# useful debugging code for checking that transforms are ok
		r = []
		x0, y0 = p
		for i, x in enumerate(range(int(x0) - size, int(x0) + size)):
			p1 = np.array([x, int(y0) - size]) * self._layout_to_page_scale
			p2 = np.array([x, int(y0) + size]) * self._layout_to_page_scale
			r.append(skimage.measure.profile_line(
				self._binarized,
				tuple(reversed(tuple(p1))),
				tuple(reversed(tuple(p2))),
				order=1, linewidth=2))
		r = np.transpose(np.array(r))
		return r

	def __call__(self, p1, p2):
		page_p1 = tuple(np.array(p1) * self._layout_to_page_scale)
		page_p2 = tuple(np.array(p2) * self._layout_to_page_scale)

		profile = skimage.measure.profile_line(
			self._binarized,
			tuple(reversed(page_p1)), tuple(reversed(page_p2)),
			order=1, linewidth=2)

		# white enough, i.e. could split here?
		return np.mean(profile) > self._black_threshold


class Squeeze:
	def __init__(self, distance, quantile, ws_profiler, cache=None):
		self._squeeze_distance = 20
		self._quantile = 0.1
		self._ws_profiler = ws_profiler
		self._cache = cache

	def __call__(self, polygon):
		lengths, paths = geometry.squeeze_paths(polygon, self._cache)

		if len(lengths) < 3:
			yield polygon
			return

		if np.quantile(lengths, self._quantile) < self._squeeze_distance:
			# uniform slim shape
			yield polygon
			return

		shortest_path = paths[np.argmin(lengths)]
		assert shortest_path[0] == "s"

		# build squeeze path.

		path = [shortest_path[1], shortest_path[-1]]

		path_length = np.linalg.norm(
			np.array(path[0]) - np.array(path[1]))

		if path_length > self._squeeze_distance:
			yield polygon
			return

		if not self._ws_profiler(path[0], path[1]):  # any obstacles?
			yield polygon
			return

		# split.

		geoms = shapely.ops.split(polygon, shapely.geometry.LineString(path))
		if not all(x.geom_type == "Polygon" for x in geoms):
			yield polygon
		else:
			for x in geoms.geoms:
				yield x


class Offset:
	def __init__(self, offset, cache=None):
		self._offset = offset
		self._cache = cache

		import skgeom as sg
		self._sg = sg

	def __call__(self, polygon):
		cache_key = ("offset", self._offset, polygon.wkt)
		if self._cache is not None and cache_key in self._cache:
			for coords in self._cache[cache_key]:
				yield shapely.geometry.Polygon(coords)
			return
		else:
			cache_data = []

		skeleton = self._sg.skeleton.create_interior_straight_skeleton(
			_shapely_to_skgeom(polygon))
		for p in skeleton.offset_polygons(self._offset):
			yield shapely.geometry.Polygon(p.coords)

			if self._cache is not None:
				cache_data.append(p.coords)

		if self._cache is not None:
			self._cache.set(cache_key, cache_data)


class EstimatePolyline:
	def __init__(self, orientation=None):
		self._factory = SkGeomMultiPolylineFactory(
			# BestPolylineFactor
			FastPolylineFactory(
				orientation=orientation,
				tolerance=0.5))

	def __call__(self, polygon):
		r = self._factory(polygon)
		if r is not None:
			yield r


class Instantiate:
	def __init__(self, class_):
		self._class = class_

	def __call__(self, polygon):
		yield self._class(polygon)


class Agglomerate:
	def __init__(self, polylines, buffer):
		self._sep = [shapely.geometry.LineString(line.coords).buffer(buffer)
			for line in polylines]
		self._sep_tree = shapely.strtree.STRtree(self._sep)

	def __call__(self, polygons):
		G = nx.Graph()

		def _connector_id(c):
			return ("sep", tuple(c.centroid.coords[0]))

		G.add_nodes_from([("blk", i) for i in range(len(polygons))])
		G.add_nodes_from([_connector_id(c) for c in self._sep])

		for i, p in enumerate(polygons):
			for c in self._sep_tree.query(p):
				G.add_edge(("blk", i), _connector_id(c))

		agglomerated = []
		for group in nx.connected_components(G):
			blks = [polygons[i] for t, i in group if t == "blk"]
			agglomerated.append(
				shapely.geometry.MultiPolygon(blks).convex_hull)

		logging.info("agglomerated %d to %d" % (len(polygons), len(agglomerated)))
		return agglomerated


class HeuristicFrameDetector:
	def __init__(self, size, width_threshold, distance_threshold, propagators):
		super().__init__()
		self._size = size
		self._width_threshold = width_threshold
		self._distance_threshold = distance_threshold
		self._propagators = propagators

	def filter(self, polygons, classes):
		w, h = self._size
		width_threshold = w * self._width_threshold
		distance_threshold = w * self._distance_threshold

		def _is_potential_noise(polygon):
			x0, y0, x1, y1 = polygon.bounds
			return x1 - x0 < width_threshold

		n_polygons = len(polygons)
		potential_noise = []

		for axis, direction in ((0, 1), (1, -1)):
			heap = []
			for i, p in enumerate(polygons):
				heappush(heap, (
					int(p.bounds[axis * 2] * direction),
					int(p.bounds[2] - p.bounds[0]),
					i, p))
			while heap and _is_potential_noise(heap[0][-1]):
				potential_noise.append(heap[0][-1])
				heappop(heap)
			polygons = [x[-1] for x in heap]

		if potential_noise and polygons:
			items = dict()
			items["frame"] = shapely.ops.unary_union(polygons).convex_hull
			for i, x in enumerate(potential_noise):
				items[("noise", i)] = x

			neighbors_ = neighbors(items)
			graph = nx.Graph()
			for a, b in neighbors_.edges():
				propagate = True
				for x in (a, b):
					if x == "frame":
						continue
					if classes[id(items[x])] not in self._propagators:
						propagate = False
						break
				if propagate and items[a].distance(items[b]) < distance_threshold:
					graph.add_edge(a, b)
			for nodes in nx.connected_components(graph):
				if "frame" in nodes:
					polygons.extend([items[x] for x in nodes if x != "frame"])
					break

		if len(polygons) < n_polygons:
			logging.info("removed %s polygons." % (n_polygons - len(polygons)))

		return polygons

	def multi_class_filter(self, polygons):
		classes = dict(itertools.chain(*[[(
			id(p), k)
			for p in class_polygons] for k, class_polygons in polygons.items()]))

		f_polygons = self.filter(
			list(itertools.chain(*list(polygons.values()))),
			classes)

		r = collections.defaultdict(list)
		for p in f_polygons:
			r[classes[id(p)]].append(p)
		return r


class Contour:
	def __init__(self, polygon):
		self._polygon = polygon
		self._coords = list(polygon.exterior.coords)
		self._pt = polygon.representative_point().coords[0]

	@property
	def coords(self):
		return self._coords
	
	@property
	def representative_point(self):
		return self._pt


def fold_operator(pipeline):
	def apply(x):
		for p in pipeline:
			x = p(x)
		return x
	return apply


def map_operator(f):
	def apply(input):
		return list(itertools.chain(*[list(f(p)) for p in input]))
	return apply


def construct(pipeline, input):
	queue = [(input, 0)]

	while queue:
		data, stage = queue.pop()
		if stage >= len(pipeline):
			yield data
		else:
			for r in pipeline[stage](data):
				if r is not None:
					queue.append((r, stage + 1))


def constructor(pipeline):
	return functools.partial(construct, pipeline)


def multi_class_constructor(pipeline, classes):

	def single_class_constructor(c):
		if isinstance(pipeline, types.LambdaType):
			return constructor(pipeline(c))
		else:
			return constructor(pipeline)

	def construct(pixels):
		if not isinstance(pixels, np.ndarray):
			pixels = np.array(pixels)

		return dict(
			(c, list(single_class_constructor(c)(pixels == c.value)))
			for c in classes)

	return construct

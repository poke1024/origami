import cv2
import skgeom as sg
import numpy as np
import networkx as nx
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.affinity
import functools
import types
import sympy
import itertools
import logging
import collections
import scipy
import skimage
import skimage.morphology
import skimage.measure
import semantic_version
import scipy.optimize
import PIL.Image
import PIL.ImageDraw

from cached_property import cached_property

import origami.core.geometry as geometry
from origami.core.skeleton import FastSkeleton
from origami.core.mask import Mask


def _without_closing_point(pts):
	if tuple(pts[0]) == tuple(pts[-1]):
		pts = pts[:-1]

	return pts


def _shapely_to_skgeom(polygon):
	pts = _without_closing_point(list(polygon.exterior.coords))
	return sg.Polygon(list(reversed(pts)))


def _skgeom_to_shapely(polygon):
	return shapely.geometry.Polygon(list(reversed(polygon.coords)))


def _as_wl(x):
	if isinstance(x, (list, tuple, np.ndarray)):
		return "{%s}" % ",".join(_as_wl(y) for y in x)
	else:
		return str(x)


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
			results.append(shapely.ops.cascaded_union([
				polygons[i] for i in names
			]).convex_hull)

	tree = shapely.strtree.STRtree(results)
	for blob in small_blobs:
		if not any(x.contains(blob) for x in tree.query(blob)):
			results.append(blob)

	return results


class Contours:
	def __init__(self, ink=None, opening=None, glue=0):
		# "ink" allows the caller to define areas that are considered
		# not connected, partially overriding the mask provided later.
		# ink shall contain connected components as they should be.
		self._ink = ink
		self._opening = opening

		self._glue = glue

	def __call__(self, mask):
		if self._ink is not None:
			ink = cv2.resize(
				self._ink.astype(np.uint8),
				tuple(reversed(mask.shape)),
				interpolation=cv2.INTER_NEAREST) > 0

			mask = np.logical_and(mask, ink)
			mask = self._opening(mask)

		polygons = []
		for pts in find_contours(mask):
			polygons.append(shapely.geometry.Polygon(pts))

		if self._glue > 0:
			glue_area = mask.size * (self._glue ** 2)
			polygons = selective_glue(polygons, glue_area)

		for polygon in polygons:
			yield polygon


class Decompose:
	def __call__(self, polygon):
		if not polygon.is_valid:
			for q in _extract_simple_polygons(
					_without_closing_point(polygon.exterior.coords)):
				q = _skgeom_to_shapely(q)
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

	def __call__(self, polygon):
		cache_key = ("offset", self._offset, polygon.wkt)
		if self._cache is not None and cache_key in self._cache:
			for coords in self._cache[cache_key]:
				yield shapely.geometry.Polygon(coords)
			return
		else:
			cache_data = []

		skeleton = sg.skeleton.create_interior_straight_skeleton(
			_shapely_to_skgeom(polygon))
		for p in skeleton.offset_polygons(self._offset):
			yield shapely.geometry.Polygon(p.coords)

			if self._cache is not None:
				cache_data.append(p.coords)

		if self._cache is not None:
			self._cache.set(cache_key, cache_data)



def _point_to_tuple(p):
	return float(p.x()), float(p.y())


def _traceback(source, backward, node):
	yield node

	while node != source:
		node = backward[node]
		yield node


def _farthest(G, source):
	farthest = (0, None)

	len_up_to = dict(((source, 0),))
	backward = dict()
	for u, v, t in nx.dfs_labeled_edges(G, source=source):
		if t == 'forward':
			if u == v:
				u_v_len = 0
			else:
				u_v_len = G[u][v]["distance"]
			path_len = len_up_to[u] + u_v_len
			len_up_to[v] = path_len

			backward[v] = u

			if path_len > farthest[0]:
				farthest = (path_len, v)

	_, node = farthest
	return list(reversed(list(_traceback(source, backward, node))))


def _leaf_nodes(G):
	degrees = np.array(list(G.degree(G.nodes)))
	return degrees[degrees[:, 1] == 1][:, 0]


def _longest_path(G):
	u = _farthest(G, _leaf_nodes(G)[0])[-1]
	return _farthest(G, u)


def _clip_path(origin, radius, path):
	path = np.array(path)
	prev_pt = origin
	while len(path) > 0:
		next_pt = path[0]
		d = np.linalg.norm(next_pt - origin)
		if d < radius:
			path = path[1:]
		else:
			intersections = sympy.intersection(
				sympy.Segment(prev_pt, next_pt),
				sympy.Circle(origin, radius))

			# intersection might not actually happen due to limited
			# fp precision in np.linalg.norm. if not, continue.

			if intersections:
				pt = intersections[0].evalf()
				origin = np.array([pt.x, pt.y], dtype=path.dtype)
				return np.vstack([[origin], path])
			else:
				path = path[1:]

	return path


def _clip_path_2(path, radius):
	path = _clip_path(path[0], radius, path[1:])

	if len(path) > 0:
		path = list(reversed(
			_clip_path(path[-1], radius, list(reversed(path[:-1])))))

	return path


def _expand_path(G, path):
	expanded_path = []
	for p, q in zip(path, path[1:]):
		cont = G[p][q]["path"]
		if expanded_path:
			while cont and cont[0] == expanded_path[-1]:
				cont = cont[1:]
		if cont:
			expanded_path.extend(cont)

	return expanded_path


class Polyline:
	def __init__(self, coords, width):
		self._coords = np.array(coords)
		self._width = width

	def affine_transform(self, matrix):
		coords = shapely.affinity.affine_transform(
			self.line_string, matrix)
		return Polyline(coords, self._width)

	@staticmethod
	def joined(lines):
		lines = [l for l in lines if l is not None]
		if not lines:
			return None
		return Polyline(
			np.vstack([l.coords for l in lines]),
			np.max([l.width for l in lines]))

	@property
	def line_string(self):
		return shapely.geometry.LineString(self.coords)

	@property
	def coords(self):
		return self._coords

	@cached_property
	def centroid(self):
		return tuple(shapely.geometry.LineString(self._coords).centroid.coords[0])

	@property
	def width(self):
		return self._width

	@property
	def is_empty(self):
		return False

	def mapped(self, m):
		pts = self._coords
		for a, b in zip(pts, pts[1:]):
			yield (m[tuple(a)], m[tuple(b)])

	def oriented(self, v):
	    u = self._coords[-1] - self._coords[0]
	    if np.dot(u, np.array(v)) < 0:
	        return Polyline(list(reversed(self._coords)), self._width)
	    else:
	        return self

	def simplify(self, tolerance):
		if len(self._coords) < 2:
			return None
		else:
			l = shapely.geometry.LineString(self._coords).simplify(tolerance)
			if not l.is_empty:
				return Polyline(l.coords, self._width)
			else:
				return None

	@property
	def segments(self):
		return list(zip(self.coords, self.coords[1:]))

	@cached_property
	def length(self):
		return sum([np.linalg.norm(b - a) for a, b in self.segments])


def _extract_simple_polygons(coords, orientation=None):
	assert coords[0] != coords[-1]

	arr = sg.arrangement.Arrangement()
	for a, b in zip(coords, coords[1:] + [coords[0]]):
		arr.insert(sg.Segment2(sg.Point2(*a), sg.Point2(*b)))

	polygons = []

	for _, boundary in geometry.face_boundaries(arr):
		polygons.append(sg.Polygon(
			list(reversed(_without_closing_point(boundary)))))

	if len(polygons) > 1 and orientation is not None:
		polygons = sorted(polygons, key=lambda p: np.dot(p.coords[0], orientation))

	return polygons


class EstimatePolyline:
	def __init__(self, orientation=None):
		self._orientation = orientation
		self._fast_skeleton = FastSkeleton()
		self._skeleton_path = dict(
			fast=self._fast_skeleton_path,
			best=self._best_skeleton_path)
		self._tolerance = 0.5
		self._quality = "fast"

	def _best_skeleton_path(self, polygon):
		try:
			skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)
		except RuntimeError as e:
			logging.error("skeleton failed on %s" % polygon)
			return None

		G = nx.Graph()
		G.add_nodes_from([_point_to_tuple(v.point) for v in skeleton.vertices])

		if len(G) < 2:
			return None, 0

		uvs = [(_point_to_tuple(h.vertex.point), _point_to_tuple(h.opposite.vertex.point))
			for h in skeleton.halfedges if h.is_bisector]

		G.add_weighted_edges_from([
			(u, v, np.linalg.norm(np.array(u) - np.array(v))) for u, v in uvs], weight="distance")

		path = _longest_path(G)
		line_width = float(max(v.time for v in skeleton.vertices))

		path = list(shapely.geometry.LineString(
			path).simplify(self._tolerance).coords)

		return np.array(path), line_width

	def _fast_skeleton_path(self, polygon):
		# need buffer of 1 to fix time computation (otherwise there might be no border background).
		mask = Mask(_skgeom_to_shapely(polygon), buffer=1)
		G = self._fast_skeleton(mask.binary, time=True)

		if len(G) < 2:
			return None, 0

		path = _longest_path(G)
		path = _expand_path(G, path)

		path = list(shapely.geometry.LineString(
			path).simplify(self._tolerance).coords)

		origin = np.array(mask.bounds[:2])
		path = [np.array(p) + origin for p in path]

		line_width = float(max(G.nodes[v]["time"] for v in G))

		return np.array(path), line_width

	def _simple_polyline(self, polygon):
		if polygon.orientation() == sg.Sign.NEGATIVE:
			logging.error("encountered negative orientation polygon")
			return None

		if polygon.orientation() != sg.Sign.POSITIVE:
			return None

		path, line_width = self._skeleton_path[self._quality](polygon)

		if path is None:
			return None

		if self._quality == "best":
			path = _clip_path_2(path, line_width)
			if not path:
				return None

		polyline = Polyline(path, line_width)

		if self._orientation is not None:
			polyline = polyline.oriented(self._orientation)

		return polyline		

	def __call__(self, polygon):
		sk_polygon = _shapely_to_skgeom(polygon)

		if not sk_polygon.is_simple():
			coords = list(polygon.exterior.coords)

			simple_polygons = _extract_simple_polygons(
				_without_closing_point(coords), self._orientation)

			if len(simple_polygons) == 0:
				# logging.error("no simple polygons")
				return
			elif len(simple_polygons) > 1:
				if self._orientation is None:
					# logging.error("multiple simple polygons without orientation")
					return

				joined = Polyline.joined(
					[self._simple_polyline(p) for p in simple_polygons])

				yield joined
			else:
				yield self._simple_polyline(simple_polygons[0])
		else:
			yield self._simple_polyline(sk_polygon)


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
	def __init__(self, size, threshold):
		super().__init__()
		self._size = size
		self._threshold = threshold

	def filter(self, polygons):
		w, h = self._size
		width_threshold = w * self._threshold

		def _is_noise(polygon):
			x0, y0, x1, y1 = polygon.bounds
			return x1 - x0 < width_threshold

		n_polygons = len(polygons)

		for i, r in ((0, False), (2, True)):
			candidates = sorted(polygons, key=lambda p: p.bounds[i], reverse=r)
			polygons = list(itertools.dropwhile(_is_noise, candidates))

		if len(polygons) < n_polygons:
			logging.info("removed %s polygons." % (n_polygons - len(polygons)))

		return polygons

	def multi_class_filter(self, polygons):
		classes = dict(itertools.chain(*[[(id(p), k)
			for p in class_polygons] for k, class_polygons in polygons.items()]))

		f_polygons = self.filter(list(itertools.chain(*list(polygons.values()))))

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

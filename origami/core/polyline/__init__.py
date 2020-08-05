import shapely
import numpy as np
import networkx as nx
import sympy
import logging

from cached_property import cached_property

from origami.core.mask import Mask
from origami.core.skeleton import FastSkeleton


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


class Polyline:
	def __init__(self, coords, width):
		self._coords = np.array(coords)
		self._width = width

	@staticmethod
	def create(path, orientation, line_width, clip_ends):
		if clip_ends:
			path = _clip_path_2(path, line_width)
			if not path:
				return None

		polyline = Polyline(path, line_width)

		if orientation is not None:
			polyline = polyline.oriented(orientation)

		return polyline

	@staticmethod
	def joined(lines):
		lines = [l for l in lines if l is not None]
		if not lines:
			return None
		return Polyline(
			np.vstack([l.coords for l in lines]),
			np.max([l.width for l in lines]))

	def affine_transform(self, matrix):
		coords = shapely.affinity.affine_transform(
			self.line_string, matrix)
		return Polyline(coords, self._width)

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
			yield m[tuple(a)], m[tuple(b)]

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


class PolylineFactory:
	def __init__(self, orientation, tolerance):
		self._orientation = orientation
		self._tolerance = tolerance

	@property
	def orientation(self):
		return self._orientation

	@property
	def tolerance(self):
		return self._tolerance

	def _longest_path(self, G):
		digraph = nx.DiGraph()
		digraph.add_nodes_from(G.nodes)

		for a, b in G.edges:
			va = np.array(a)
			vb = np.array(b)
			xa = np.dot(va, self._orientation)
			xb = np.dot(vb, self._orientation)
			d = np.linalg.norm(va - vb)
			if xa < xb:
				digraph.add_edge(a, b, distance=d)
			elif xa > xb:
				digraph.add_edge(b, a, distance=d)
			else:
				pass  # do not break DAG by adding edge

		return nx.algorithms.dag.dag_longest_path(
			digraph, weight="distance")

	def _expand_path(self, G, path):
		expanded_path = []
		for p, q in zip(path, path[1:]):
			cont = G[p][q]["path"]
			if expanded_path:
				while cont and cont[0] == expanded_path[-1]:
					cont = cont[1:]
			if cont:
				expanded_path.extend(cont)

		return expanded_path


class MultiPolylineFactory:
	def __init__(self):
		pass


class FastPolylineFactory(PolylineFactory):
	def __init__(self, orientation, tolerance):
		super().__init__(orientation, tolerance)
		self._fast_skeleton = FastSkeleton()

	def __call__(self, polygon):
		# need buffer of 1 to fix time computation
		# (otherwise there might be no border background).
		mask = Mask(polygon, buffer=1)
		G = self._fast_skeleton(mask.binary, time=True)

		if len(G) < 2:
			return None

		path = self._longest_path(G)
		path = self._expand_path(G, path)

		simplified = shapely.geometry.LineString(
			path).simplify(self.tolerance)
		if simplified.is_empty:
			return None

		if simplified.geom_type != "LineString":
			logging.warning("unexpected line geometry %s" % simplified.geom_type)

		path = list(simplified.coords)

		origin = np.array(mask.bounds[:2])
		path = [np.array(p) + origin for p in path]

		line_width = float(max(G.nodes[v]["time"] for v in G))

		return Polyline.create(
			path, self.orientation, line_width, clip_ends=False)

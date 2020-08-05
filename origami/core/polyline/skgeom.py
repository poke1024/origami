import shapely
import numpy as np
import networkx as nx
import skgeom as sg
import logging

import origami.core.geometry as geometry
from origami.core.polyline import Polyline, PolylineFactory, MultiPolylineFactory


def _without_closing_point(pts):
	if tuple(pts[0]) == tuple(pts[-1]):
		pts = pts[:-1]

	return pts


def _point_to_tuple(p):
	return float(p.x()), float(p.y())


def _shapely_to_skgeom(polygon):
	pts = _without_closing_point(list(polygon.exterior.coords))
	return sg.Polygon(list(reversed(pts)))


def _skgeom_to_shapely(polygon):
	return shapely.geometry.Polygon(list(reversed(polygon.coords)))


def _extract_simple_polygons_skgeom(coords, orientation=None):
	coords = _without_closing_point(coords)

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


def extract_simple_polygons(coords, orientation=None):
	return [_skgeom_to_shapely(p) for p in _extract_simple_polygons_skgeom(coords, orientation)]


class SkGeomMultiPolylineFactory(MultiPolylineFactory):
	def __init__(self, factory):
		self._factory = factory

	@property
	def orientation(self):
		return self._factory.orientation

	def _simple_polyline(self, polygon):
		if polygon.orientation() == sg.Sign.NEGATIVE:
			logging.error("encountered negative orientation polygon")
			return None

		if polygon.orientation() != sg.Sign.POSITIVE:
			return None

		return self._factory(_skgeom_to_shapely(polygon))

	def __call__(self, polygon):
		sk_polygon = _shapely_to_skgeom(polygon)

		if not sk_polygon.is_simple():
			coords = list(polygon.exterior.coords)

			simple_polygons = _extract_simple_polygons_skgeom(
				coords, self.orientation)

			if len(simple_polygons) == 0:
				# logging.error("no simple polygons")
				return None
			elif len(simple_polygons) > 1:
				if self.orientation is None:
					# logging.error("multiple simple polygons without orientation")
					return None

				joined = Polyline.joined(
					[self._simple_polyline(p) for p in simple_polygons])

				return joined
			else:
				return self._simple_polyline(simple_polygons[0])
		else:
			return self._simple_polyline(sk_polygon)


class BestPolylineFactory(PolylineFactory):
	def __init__(self, orientation, tolerance):
		super().__init__(orientation, tolerance)

	def __call_(self, polygon):
		polygon = _shapely_to_skgeom(polygon)

		try:
			skeleton = sg.skeleton.create_interior_straight_skeleton(polygon)
		except RuntimeError as e:
			logging.error("skeleton failed on %s" % polygon)
			return None

		G = nx.Graph()
		G.add_nodes_from([_point_to_tuple(v.point) for v in skeleton.vertices])

		if len(G) < 2:
			return None

		uvs = [
			(_point_to_tuple(h.vertex.point), _point_to_tuple(h.opposite.vertex.point))
			for h in skeleton.halfedges if h.is_bisector]

		G.add_weighted_edges_from([
			(u, v, np.linalg.norm(np.array(u) - np.array(v))) for u, v in uvs], weight="distance")

		path = self._longest_path(G)
		line_width = float(max(v.time for v in skeleton.vertices))

		path = list(shapely.geometry.LineString(
			path).simplify(self.tolerance).coords)

		return Polyline.create(
			path, self.orientation, line_width, clip_ends=True)

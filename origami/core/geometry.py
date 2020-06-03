import enum
import math
import logging

import numpy as np
import skgeom as sg
import networkx as nx

import shapely
import shapely.ops

import scipy.ndimage


def _tuple(p):
	return (float(p.x()), float(p.y()))


def _shapely_to_skgeom(polygon):
	pts = _without_closing_point(list(polygon.exterior.coords))
	return sg.Polygon(list(reversed(pts)))


def _without_closing_point(pts):
	if tuple(pts[0]) == tuple(pts[-1]):
		pts = pts[:-1]

	return pts


class Margin(enum.Enum):
	TOP = 0
	LEFT = 1
	BOTTOM = 2
	RIGHT = 3

	def is_adjacent_to(self, other):
		return abs(self.value - other.value) <= 1


def _masked_extract(items, mask):
	items = np.array(items)

	i = np.nonzero(mask)[0]
	if i.size == 0:
		return np.array([])

	if i[0] == 0 and i[-1] == len(mask) - 1:  # wrap?
		j = np.nonzero(np.logical_not(mask))[0]
		if j.size == 0:
			return items[i]
		else:
			return np.concatenate([items[j[-1] + 1:], items[:j[0]]], axis=0)
	else:
		return items[np.min(i):np.max(i)]


def _extract_margins(pts, annotated, dilate=True):
	pts = np.array(pts)

	margins = dict()
	for m in Margin:
		mask = np.array([m in x for x in annotated], np.bool)

		if dilate:
			mask = scipy.ndimage.morphology.binary_dilation(mask)

		r = _masked_extract(pts, mask)
		if r.size > 0:
			margins[m] = [tuple(x) for x in r]

	return margins


def set_euclidean_weights(graph):
	def _euclidean(a, b):
		return float(np.linalg.norm(np.array(a) - np.array(b)))

	nx.set_edge_attributes(
		graph,
		dict(((a, b), _euclidean(a, b)) for a, b in graph.edges),
		'euclidean')


def embedded_graph(nodes, edges):
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	set_euclidean_weights(G)
	return G


def compute_graph_margins(G):
	# (1) pick the graph M component that contains the corner nodes.
	# (2) in M, compute an arrangement to find unbounded faces.
	# (3) from these faces' edges, computer an outer graph.
	# (4) on this outer graph, compute margins by shortest paths.

	assert not nx.is_directed(G)

	components = list(nx.connected_components(G))
	assert len(components) == 1

	node00 = sorted(list(G.nodes), key=lambda node: (node[0], node[1]))[0]
	node10 = sorted(list(G.nodes), key=lambda node: (-node[0], node[1]))[0]
	node11 = sorted(list(G.nodes), key=lambda node: (-node[0], -node[1]))[0]
	node01 = sorted(list(G.nodes), key=lambda node: (node[0], -node[1]))[0]
	corners = (node00, node10, node11, node01)

	main_graph = G.subgraph(
		[nodes for nodes in components if node00 in nodes][0])
	if not all(c in main_graph for c in corners):
		#print("corners", corners)
		#print("G", list(G.edges))
		raise ValueError()

	arr = sg.arrangement.Arrangement()

	for a, b in main_graph.edges:
		arr.insert(sg.Segment2(sg.Point2(*a), sg.Point2(*b)))

	boundary = set()

	for h in arr.halfedges:
		if h.face().is_unbounded():
			c = h.curve()
			boundary.add(_tuple(c.source()))
			boundary.add(_tuple(c.target()))

	outer_graph = main_graph.subgraph(boundary)
	assert all(c in outer_graph for c in corners)

	set_euclidean_weights(outer_graph)

	return dict((
		(Margin.LEFT, nx.shortest_path(outer_graph, node00, node01, "euclidean")),
		(Margin.RIGHT, nx.shortest_path(outer_graph, node10, node11, "euclidean")),
		(Margin.TOP, nx.shortest_path(outer_graph, node00, node10, "euclidean")),
		(Margin.BOTTOM, nx.shortest_path(outer_graph, node01, node11, "euclidean"))))


def closed_boundary(p):
	if p[0] != p[-1]:
		return zip(p, p[1:] + [p[0]])
	else:
		return zip(p, p[1:])


def _maximize_margins_area(pts, corner_pts):
	def _area(corners):
		return shapely.geometry.Polygon([pts[i] for i in corners]).area

	def _variants(corners, k):
		i = corners[k]

		for j in (
			i,
			(i + len(pts) - 1) % len(pts),
			(i + 1) % len(pts)):

			modified_corners = corners.copy()
			modified_corners[k] = j
			yield modified_corners

	pt_index = dict((tuple(p), i) for i, p in enumerate(pts))
	corners = np.array([pt_index[tuple(p)] for p in corner_pts])

	for k in range(4):
		while True:
			n_corners = max(_variants(corners, k), key=_area)
			if np.array_equal(n_corners, corners):
				break
			corners = n_corners

	return [tuple(pts[i]) for i in corners]


def compute_margins_from_boundary(boundary_pts, phi=-math.pi / 2, cache=None):

	k_gon = sg.maximum_area_inscribed_k_gon(
		[sg.Point2(*p) for p in boundary_pts],
		4)

	corners = [(float(v.x()), float(v.y())) for v in k_gon.vertices]

	def _sort_by(a, axis):
		return sorted(a, key=lambda p: p[axis])

	by_y = _sort_by(corners, 1)

	top_left, top_right = _sort_by(by_y[:2], 0)
	bottom_left, bottom_right = _sort_by(by_y[2:], 0)

	top_left, top_right, bottom_right, bottom_left = _maximize_margins_area(
		boundary_pts, [
			top_left,
			top_right,
			bottom_right,
			bottom_left])

	pts = [tuple(x) for x in boundary_pts]
	G = nx.Graph()
	G.add_nodes_from(pts)
	G.add_edges_from(list(closed_boundary(pts)))
	set_euclidean_weights(G)

	m = dict()
	m[Margin.TOP] = nx.shortest_path(G, top_left, top_right, weight='euclidean')
	m[Margin.RIGHT] = nx.shortest_path(G, top_right, bottom_right, weight='euclidean')
	m[Margin.BOTTOM] = nx.shortest_path(G, bottom_right, bottom_left, weight='euclidean')
	m[Margin.LEFT] = nx.shortest_path(G, bottom_left, top_left, weight='euclidean')
	return m

	if False:
		annotated = [[] for _ in range(len(boundary_pts))]
	
		pts = np.array(boundary_pts)

		xmin = np.min(pts[:, 0])
		xmax = np.max(pts[:, 0])
		ymin = np.min(pts[:, 1])
		ymax = np.max(pts[:, 1])

		largest = sg.LargestEmptyIsoRectangle(
		    sg.Point2(xmin, ymin), sg.Point2(xmax, ymax))
		largest.insert([sg.Point2(*p) for p in convex_boundary_pts])
		rectangle = largest.largest_empty_iso_rectangle

		r_xmin = rectangle.xmin()
		r_xmax = rectangle.xmax()
		r_ymin = rectangle.ymin()
		r_ymax = rectangle.ymax()

		eps = 0.5

		for m, p in zip(annotated, boundary_pts):
			if p[0] >= r_xmax - eps:
				m.append(Margin.RIGHT)
			if p[0] <= r_xmin + eps:
				m.append(Margin.LEFT)
			if p[1] >= r_ymax - eps:
				m.append(Margin.BOTTOM)
			if p[1] <= r_ymin + eps:
				m.append(Margin.TOP)

		return _extract_margins(boundary_pts, annotated)


def squeeze_paths(polygon, cache=None):
	cache_key = ("squeeze-paths", polygon.wkt)
	if cache is not None and cache_key in cache:
		lengths, paths = cache[cache_key]
		return np.array(lengths), paths

	margins = compute_margins_from_boundary(
		list(reversed(polygon.exterior.coords)))

	try:
		top = margins[Margin.TOP]
		bottom = margins[Margin.BOTTOM]
	except KeyError:
		if cache is not None:
			cache.set(cache_key, ([], []))
		return [], []

	G = nx.Graph()
	G.add_nodes_from([tuple(x) for x in polygon.exterior.coords])

	skeleton = sg.skeleton.create_interior_straight_skeleton(
		_shapely_to_skgeom(polygon))

	for h in skeleton.halfedges:
		if h.is_bisector:
			a = _tuple(h.vertex.point)
			b = _tuple(h.opposite.vertex.point)
			G.add_node(a)
			G.add_node(b)
			G.add_edge(a, b, weight=np.linalg.norm(np.array(a) - np.array(b)))

	G.add_node("s")
	for x in top:
		G.add_edge("s", tuple(x), weight=0)

	lengths, paths = nx.single_source_dijkstra(
		G, source="s", target=None, weight="weight")

	lengths = [lengths[tuple(x)] for x in bottom]
	paths = [paths[tuple(x)] for x in bottom]

	if cache is not None:
		cache.set(cache_key, (lengths, paths))

	return np.array(lengths), paths


def face_boundaries(arr):
	for f in arr.faces:
		if not f.has_outer_ccb():
			continue

		h0 = None
		segments = []
		for h in f.outer_ccb:
			if h0 is None:
				h0 = h
			elif h is h0:
				break
			curve = h.curve()
			segments.append((
				_tuple(curve.source()),
				_tuple(curve.target())))

		line = shapely.ops.linemerge(segments)
		if line.geom_type != "LineString":
			logging.error("line.geom_type == %s" % line.geom_type)
			continue

		face_coords = _without_closing_point(line.coords)
		if len(face_coords) < 3:
			continue

		if shapely.geometry.Polygon(face_coords).exterior.is_ccw:
			face_coords = list(reversed(face_coords))

		yield f, face_coords

import pyvoronoi
import networkx as nx
import numpy as np
import shapely.strtree


def shape_collection_bounds(shapes, margin=0):
	non_empty = [s.bounds for s in shapes if not s.is_empty]
	bounds = np.array(non_empty).reshape((len(non_empty), 2, 2))
	minx = np.min(bounds[:, 0, 0])
	miny = np.min(bounds[:, 0, 1])
	maxx = np.max(bounds[:, 1, 0])
	maxy = np.max(bounds[:, 1, 1])
	return minx - margin, miny - margin, maxx + margin, maxy + margin


def non_overlapping_shapes(shapes):
	graph = nx.Graph()
	graph.add_nodes_from([id(x) for x in shapes])
	lookup = dict((id(x), x) for x in shapes)
	index = dict((id(x), i) for i, x in enumerate(shapes))

	tree = shapely.strtree.STRtree(shapes)
	for shape in shapes:
		for t in tree.query(shape):
			if t is shape:
				continue
			if not t.intersects(shape):
				continue
			if t.touches(shape):
				continue
			graph.add_edge(id(shape), id(t))

	safe_shapes = []
	mapping = []
	for xs in nx.connected_components(graph):
		if len(xs) == 1:
			safe_shapes.append(lookup[list(xs)[0]])
		else:
			u = shapely.ops.cascaded_union([
				lookup[x] for x in xs])
			assert u.geom_type == "Polygon"
			safe_shapes.append(u)
		mapping.append([index[x] for x in xs])

	return safe_shapes, mapping


def unsafe_neighbors(shapes, scale=100, border=1):
	# shapes must not intersect.

	pv = pyvoronoi.Pyvoronoi(scale)
	site_to_shape = []

	for i, shape in enumerate(shapes):
		if shape.is_empty:
			continue
		coords = np.asarray(shape.exterior)
		assert np.all(coords[0] == coords[-1])
		for a, b in zip(coords, coords[1:]):
			assert np.any(a != b)
			pv.AddSegment((a, b))
			site_to_shape.append(i)

	minx, miny, maxx, maxy = shape_collection_bounds(
		shapes, border)
	pv.AddSegment(((minx, miny), (maxx, miny)))
	pv.AddSegment(((maxx, miny), (maxx, maxy)))
	pv.AddSegment(((maxx, maxy), (minx, maxy)))
	pv.AddSegment(((minx, maxy), (minx, miny)))
	site_to_shape.extend([-1, -1, -1, -1])

	pv.Construct()

	edges = pv.GetEdges()
	cells = pv.GetCells()

	graph = nx.Graph()
	graph.add_nodes_from(range(len(cells)))

	def adjacent_cell():
		for edge in edges:
			yield edge.cell, edges[edge.twin].cell

	graph.add_edges_from(adjacent_cell())

	for i, cell in enumerate(cells):
		if cell.is_open or cell.site < 0:
			graph.remove_node(i)
		elif not cell.contains_segment:
			edges = np.array(list(graph.edges(i)))
			nhood = set(edges.flatten()) - set([i])
			nhood = list(nhood)
			for j, x in enumerate(nhood):
				for y in nhood[j + 1:]:
					graph.add_edge(x, y)

			graph.remove_node(i)

	partitions = [[] for _ in shapes]
	for i in graph.nodes():
		k = site_to_shape[cells[i].site]
		partitions[k].append(i)

	graph = nx.quotient_graph(graph, partitions)

	mapping = dict()
	for group in graph.nodes:
		if group:
			mapping[group] = site_to_shape[
				cells[next(iter(group))].site]
	graph = nx.relabel_nodes(graph, mapping)

	return graph


def indexed_neighbors(shapes, simplify=100, **kwargs):
	# neighborhood is defined here as a path between
	# two regions that does not cross an influence
	# (i.e. nearest) region of a third region.

	# simplify, get rid of intersections.
	shapes = [shape.simplify(simplify) for shape in shapes]

	shapes_t, mapping_t = non_overlapping_shapes(shapes)
	graph_t = unsafe_neighbors(shapes_t, **kwargs)

	graph = nx.Graph()
	graph.add_nodes_from(range(len(shapes)))

	for a_t, b_t in graph_t.edges():
		for x in mapping_t[a_t]:
			for y in mapping_t[b_t]:
				graph.add_edge(x, y)

	for group in mapping_t:
		for i, x in enumerate(group):
			for y in group[i + 1:]:
				graph.add_edge(x, y)

	return graph


def neighbors(named_shapes, **kwargs):
	named_shapes = list(named_shapes.items())

	shapes = [v for _, v in named_shapes]
	names = [k for k, _ in named_shapes]
	mapping = dict((i, name) for i, name in enumerate(names))

	return nx.relabel_nodes(
		indexed_neighbors(shapes, **kwargs), mapping)

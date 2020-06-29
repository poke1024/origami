import imghdr
import numpy as np
import json
import click
import shapely.ops
import shapely.wkt
import shapely.strtree
import shapely.geometry
import shapely.ops
import sklearn.cluster
import skimage.filters
import scipy.spatial
import zipfile
import PIL.Image
import networkx
import collections
import cv2
import intervaltree
import portion

from pathlib import Path
from atomicwrites import atomic_write
from functools import partial, reduce

from origami.batch.core.block_processor import BlockProcessor
from origami.core.dewarp import Dewarper, Grid
from origami.core.predict import PredictorType
from origami.core.segment import Segmentation
from origami.core.separate import Separators
from origami.core.math import inset_bounds


def overlap_ratio(a, b):
	if a.area > b.area:
		a, b = b, a
	return a.intersection(b).area / a.area


def create_filter(s):
	good = set([tuple(s.split("/"))])
	return lambda k: k[:2] in good


def fixed_point(func, x0, reduce):
	while True:
		x = func(x0)
		if reduce(x) == reduce(x0):
			return x
		x0 = x


def _cohesion(shapes, union):
	return sum([shape.area for shape in shapes]) / union.area


class LineCounts:
	def __init__(self, lines):
		num_lines = collections.defaultdict(int)
		for path, line in lines.items():
			num_lines[path[:3]] += 1
		self._num_lines = num_lines

	def add(self, name, count):
		self._num_lines[name] = count

	def remove(self, name):
		del self._num_lines[name]

	def combine(self, names, target):
		self._num_lines[target] = sum([
			self._num_lines[x] for x in names
		])

	def __getitem__(self, block_path):
		return self._num_lines[block_path]


class Regions:
	def __init__(self, page, lines, contours, separators, union):
		self._page = page

		self._contours = dict(contours)
		self._unmodified_contours = self._contours.copy()
		for k, contour in contours:
			contour.name = "/".join(k)
		self.separators = separators

		self._line_counts = LineCounts(lines)
		self._union = union

		max_labels = collections.defaultdict(int)
		for k in self._contours.keys():
			max_labels[k[:2]] = max(max_labels[k[:2]], int(k[2]))
		self._max_labels = max_labels

	@property
	def page(self):
		return self._page

	def union(self, shapes):
		return self._union(self._page, shapes)

	@property
	def unmodified_contours(self) -> dict:
		return self._unmodified_contours

	@property
	def contours(self) -> dict:
		return self._contours

	@property
	def by_labels(self):
		by_labels = collections.defaultdict(list)
		for k, contour in self._contours.items():
			by_labels[k[:2]].append(contour)
		return by_labels

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

	def combine(self, sources):
		agg_path = min(sources)

		u = self.union([self._contours[p] for p in sources])
		self.modify_contour(agg_path, u)
		self._line_counts.combine(sources, agg_path)

		for k in sources:
			if k != agg_path:
				self.remove_contour(k)

	def combine_from_graph(self, graph):
		if graph.number_of_edges() > 0:
			for nodes in networkx.connected_components(graph):
				self.combine(nodes)

	def modify_contour(self, path, contour):
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


class Transformer:
	def __init__(self, operators):
		self._operators = operators

	def __call__(self, regions):
		for operator in self._operators:
			operator(regions)


class Dilation:
	def __init__(self):
		pass

	def __call__(self, regions):
		regions.map(lambda _, contour: regions.union([contour]))


class AdjacencyMerger:
	def __init__(self, filters, max_line_count=3, cohesion=0.8, alignment=0.8):
		self._filter = create_filter(filters)
		self._max_line_count = max_line_count
		self._cohesion = cohesion
		self._min_alignment = alignment

	def _should_merge(self, regions, p, q):
		lc = regions.line_count
		if max(lc(p), lc(q)) > self._max_line_count:
			return False

		contours = regions.contours
		a = contours[p]
		b = contours[q]

		_, ay0, _, ay1 = a.bounds
		_, by0, _, by1 = b.bounds

		shared = portion.closed(ay0, ay1) & portion.closed(by0, by1)
		if shared.empty:
			return False

		overlap = shared.upper - shared.lower
		alignment = overlap / min(ay1 - ay0, by1 - by0)
		if alignment < self._min_alignment:
			return False

		u = regions.union([a, b])
		c = _cohesion([a, b], u)
		return c > self._cohesion

	def __call__(self, regions):
		paths = []
		points = []
		for k, c in regions.contours.items():
			minx, miny, maxx, maxy = c.bounds
			pts = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
			for pt in pts:
				paths.append(k)
				points.append(pt)

		vor = scipy.spatial.Voronoi(points)

		neighbors = collections.defaultdict(set)
		for i, j in vor.ridge_points:
			if not self._filter(paths[i]):
				continue
			if paths[i][:2] == paths[j][:2]:
				neighbors[paths[i]].add(paths[j])
				neighbors[paths[j]].add(paths[i])

		graph = networkx.Graph()
		graph.add_nodes_from(paths)

		checked = set()
		for p in paths:
			for q in neighbors[p]:
				if (p, q) in checked:
					continue
				if self._should_merge(regions, p, q):
					graph.add_edge(p, q)
				checked.add((p, q))

		regions.combine_from_graph(graph)


class OverlapMerger:
	def __init__(self, maximum_overlap):
		self._maximum_overlap = maximum_overlap

	def _merge(self, regions, contours):
		graph = networkx.Graph()
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

		regions.combine_from_graph(graph)

		return graph.number_of_edges() > 0

	def __call__(self, regions):
		modify = set(regions.by_labels.keys())
		while modify:
			changed = set()
			for k, contours in regions.by_labels.items():
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
		self._shape = shapely.ops.cascaded_union(polygons)

	def __call__(self, shape):
		error_area = self._shape.intersection(shape).area
		return error_area / shape.area


class UnionOperator:
	def __init__(self, concavity, detail):
		if concavity != "bounds":
			try:
				concavity = float(concavity)
			except:
				raise click.BadOptionUsage(
					"region-concavity", "must be numeric or 'bounds'")
		self._concavity = concavity
		self._detail = detail

	def __call__(self, page, shapes):
		if len(shapes) > 1:
			polygon = shapely.ops.cascaded_union(shapes)
		else:
			polygon = shapes[0]

		concavity = self._concavity

		if concavity == "bounds":
			return shapely.geometry.box(*polygon.bounds)
		elif concavity > 1:
			from origami.concaveman import concaveman2d

			mag = self._page.magnitude(dewarped=True)
			ext = np.array(polygon.exterior.coords)
			pts = concaveman2d(
				ext,
				scipy.spatial.ConvexHull(ext).vertices,
				concavity=concavity,
				lengthThreshold=mag * self._detail)
			return shapely.geometry.Polygon(pts)
		else:  # disable concaveman
			return polygon


class SequentialMerger:
	def __init__(self, filters, cohesion, max_error, fringe, obstacles):
		self._filter = create_filter(filters)
		self._cohesion = cohesion
		self._max_error = max_error
		self._fringe = fringe
		self._obstacles = obstacles

	def _merge(self, regions, names, error_overlap):
		contours = regions.contours
		shapes = [contours[x] for x in names]

		mag = regions.page.magnitude(dewarped=True)
		fringe = self._fringe * mag
		label = names[0][:2]
		assert(all(x[:2] == label for x in names))

		graph = networkx.Graph()
		graph.add_nodes_from(names)

		i = 0
		while i < len(shapes):
			good = False
			for j in range(i + 1, len(shapes)):
				u = regions.union(shapes[i:j + 1])

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

		regions.combine_from_graph(graph)

	def _compute_order(self, regions, contours):
		mag = regions.page.magnitude(dewarped=True)
		fringe = self._fringe * mag
		contours = [(tuple(c.name.split("/")), c) for c in contours]
		return polygon_order(contours, fringe=fringe)

	def __call__(self, regions):
		by_labels = regions.by_labels
		labels = set(by_labels.keys())

		for path, contours in by_labels.items():
			if not self._filter(path):
				continue

			order = self._compute_order(regions, contours)

			error_overlap = Overlap(
				regions.unmodified_contours,
				labels - set([path[:2]]))

			self._merge(
				regions,
				order,
				error_overlap)


class Shrinker:
	def __init__(self):
		pass

	def __call__(self, regions):
		by_labels_nomod = collections.defaultdict(list)
		for k, contour in regions.unmodified_contours.items():
			by_labels_nomod[k[:2]].append(contour)

		for k0, v0 in by_labels_nomod.items():
			tree = shapely.strtree.STRtree(v0)
			for k, contour in regions.contours.items():
				if k[:2] != k0[:2]:
					continue
				try:
					q = tree.query(contour)
					if q:
						bounds = shapely.ops.cascaded_union(q).bounds
						box = shapely.geometry.box(*bounds)
						regions.modify_contour(
							k, box.intersection(contour))
				except ValueError:
					pass  # deformed geometry errors


class FixSpillOver:
	def __init__(self, filters, band=0.01, peak=0.9, min_line_count=3, window_size=15):
		self._filter = create_filter(filters)
		self._band = band
		self._peak = peak
		self._min_line_count = min_line_count
		self._window_size = window_size

	def __call__(self, regions):
		# for each contour, sample the binarized image.
		# if we detect a whitespace region in a column,
		# we split the polygon here and now.
		# since we dewarped, we know columns are unskewed.
		page = regions.page
		pixels = np.array(page.dewarped.convert("L"))

		kernel_w = max(
			10,  # pixels in warped image space
			int(np.ceil(page.magnitude(False) * self._band)))

		splits = []

		for k, contour in regions.contours.items():
			if not self._filter(k):
				continue

			if regions.line_count(k) < self._min_line_count:
				continue

			minx, miny, maxx, maxy = contour.bounds

			miny = int(max(0, miny))
			minx = int(max(0, minx))
			maxy = int(min(maxy, pixels.shape[0]))
			maxx = int(min(maxx, pixels.shape[1]))

			crop = pixels[miny:maxy, minx:maxx]

			thresh_sauvola = skimage.filters.threshold_sauvola(
				crop, window_size=self._window_size)
			crop = (crop > thresh_sauvola).astype(np.uint8)

			whitespace = np.mean(crop.astype(np.float32), axis=0)
			whitespace = np.convolve(
				whitespace, np.ones((kernel_w,)) / kernel_w, mode="same")

			peaks, info = scipy.signal.find_peaks(
				whitespace, height=self._peak)

			if len(peaks) > 0:
				i = np.argmax(info["peak_heights"])
				x = peaks[i] + minx

				sep = shapely.geometry.LineString([
					[x, -1], [x, pixels.shape[0] + 1]
				])

				splits.append((k, contour, sep))

		for k, contour, sep in splits:
			regions.remove_contour(k)
			for shape in shapely.ops.split(contour, sep).geoms:
				regions.add_contour(k[:2], shape)


class TableLayoutDetector:
	def __init__(self, filters, column_label, min_column_distance=50):
		self._filter = create_filter(filters)
		self._column_label = column_label
		self._min_column_distance = min_column_distance

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

		for sep in regions.separators.for_label(self._column_label):
			for contour in tree.query(sep):
				if contour.intersects(sep):
					path = tuple(contour.name.split("/"))
					mx = np.median(np.array(sep.coords)[:, 0])
					seps[path].append(mx)

		agg = sklearn.cluster.AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=self._min_column_distance,
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

		return dict(
			version=1,
			columns=dict(
				("/".join(path), [round(x, 1) for x in xs])
				for path, xs in columns.items()))


class LayoutDetectionProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._overwrite = self._options["overwrite"]

		self._union = UnionOperator(
			self._options["region_concavity"],
			self._options["region_detail"]
		)

		# bbz specific settings.

		self._transformer = Transformer([
			Dilation(),
			AdjacencyMerger("regions/TEXT", max_line_count=3),
			OverlapMerger(self._options["maximum_overlap"]),
			Shrinker(),
			SequentialMerger(
				filters="regions/TABULAR",
				cohesion=(0.5, 0.8),
				max_error=0.05,
				fringe=self._options["fringe"],
				obstacles=[
					"separators/H",
					"separators/V"]),
			FixSpillOver("regions/TEXT")
		])

		self._table_layout_detector = TableLayoutDetector(
			"regions/TABULAR", "separators/T")

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".dewarped.contours.zip").exists() and (
				self._overwrite or (
					not p.with_suffix(".aggregate.contours.zip").exists()))

	def process(self, page_path: Path):
		blocks = self.read_dewarped_blocks(page_path)

		if not blocks:
			return

		segmentation = Segmentation.open(
			page_path.with_suffix(".segment.zip"))
		separators = Separators(
			segmentation, self.read_dewarped_separators(page_path))

		warped_blocks = self.read_blocks(page_path)
		warped_lines = self.read_lines(page_path, warped_blocks)

		zf_path = page_path.with_suffix(".dewarped.contours.zip")
		with zipfile.ZipFile(zf_path, "r") as zf:
			meta = zf.read("meta.json")

		page = list(blocks.values())[0].page
		contours = [(k, block.image_space_polygon) for k, block in blocks.items()]

		regions = Regions(
			page, warped_lines,
			contours, separators,
			self._union)
		self._transformer(regions)

		table_data = self._table_layout_detector(regions)

		with atomic_write(
			page_path.with_suffix(".tables.json"),
			mode="wb", overwrite=self._overwrite) as f:
			f.write(json.dumps(table_data).encode("utf8"))

		zf_path = page_path.with_suffix(".aggregate.contours.zip")
		with atomic_write(zf_path, mode="wb", overwrite=self._overwrite) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				zf.writestr("meta.json", meta)
				for path, shape in regions.contours.items():
					zf.writestr("/".join(path) + ".wkt", shape.wkt.encode("utf8"))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--region-concavity',
	type=str,
	default="bounds")
@click.option(
	'--region-detail',
	type=float,
	default=0.01)
@click.option(
	'--maximum-overlap',
	type=float,
	default=0.1)
@click.option(
	'--fringe',
	type=float,
	default=0.001)
@click.option(
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
		 "but is necessary on some network file systems.")
@click.option(
	'--overwrite',
	is_flag=True,
	default=False,
	help="Recompute and overwrite existing result files.")
def detect_layout(data_path, **kwargs):
	""" Detect layout and reading order for documents in DATA_PATH. """
	processor = LayoutDetectionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	detect_layout()

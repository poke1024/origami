import imghdr
import numpy as np
import json
import click
import shapely.ops
import shapely.wkt
import zipfile
import PIL.Image
import shapely.strtree
import shapely.geometry
import shapely.ops
import sklearn.cluster
import networkx
import collections

from pathlib import Path
from atomicwrites import atomic_write
from functools import partial

from origami.batch.core.block_processor import BlockProcessor
from origami.core.dewarp import Dewarper, Grid
from origami.core.predict import PredictorType
from origami.core.xycut import polygon_order
from origami.core.segment import Segmentation


def overlap_ratio(a, b):
	if a.area > b.area:
		a, b = b, a
	return a.intersection(b).area / a.area


def aggregation(contours, union, threshold, combine):
	graph = networkx.Graph()
	graph.add_nodes_from([c.name for c in contours])

	tree = shapely.strtree.STRtree(contours)
	for contour in contours:
		for other in tree.query(contour):
			if contour.name == other.name:
				continue
			if overlap_ratio(contour, other) > threshold:
				graph.add_edge(contour.name, other.name)

	results = []
	by_name = dict((c.name, c) for c in contours)
	for names in networkx.connected_components(graph):
		target = min(names)
		combine(names, target)
		shapes = [by_name[name] for name in names]
		shape = union(shapes)
		shape.name = target
		results.append(shape)

	return results


def fixed_point(func, x0, reduce):
	while True:
		x = func(x0)
		if reduce(x) == reduce(x0):
			return x
		x0 = x


def _cohesion(shapes, union):
	return sum([shape.area for shape in shapes]) / union.area


class SequentialMerger:
	def __init__(self, cohesion, max_error, line_limit):
		self._cohesion = cohesion
		self._max_error = max_error
		self._line_limit = line_limit

	def __call__(self, names, shapes, union, error_overlap, line_counts):
		i = 0
		while i < len(shapes):
			good = False
			for j in range(i + 1, len(shapes)):
				if self._line_limit is not None:
					if max(line_counts[x] for x in names[i:j + 1]) > self._line_limit:
						break

				u = union(shapes[i:j + 1])

				cohesion = _cohesion(shapes[i:j + 1], u)
				error = error_overlap(u)

				if cohesion < self._cohesion[0] or error > self._max_error:
					break
				elif cohesion > self._cohesion[1]:
					shapes[j] = u
					i = j
					good = True
					break

			if not good:
				yield names[i], shapes[i]
				i += 1


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


class LineCounts:
	def __init__(self, lines):
		num_lines = collections.defaultdict(int)
		for path, line in lines.items():
			num_lines[path[:3]] += 1
		self._num_lines = num_lines

	def combine(self, names, target):
		self._num_lines[target] = sum([
			self._num_lines[x] for x in names
		])

	def __getitem__(self, block_path):
		return self._num_lines[block_path]


class Separators:
	def __init__(self, page_path, seps):
		segmentation = Segmentation.open(
			page_path.with_suffix(".segment.zip"))

		self._predictions = dict()
		for p in segmentation.predictions:
			if p.type == PredictorType.SEPARATOR:
				self._predictions[p.name] = p

		parsed_seps = collections.defaultdict(list)
		for k, geom in seps.items():
			prediction_name, prediction_type = k[:2]
			prediction = self._predictions[prediction_name]
			parsed_seps[prediction.classes[prediction_type]].append(geom)

		self._seps = parsed_seps

		# bbz specific.
		self._table_columns = self._seps[
			self._predictions["separators"].classes["T"]]

		self._min_column_distance = 50

	def sample(self, bounds):  # dewarped image space
		pass

	def detect_table_data(self, contours):
		contours = dict([
			(k, v) for k, v in contours.items()
			if tuple(k[:2]) == ("regions", "TABULAR")])

		for k, contour in contours.items():
			contour.name = "/".join(k)

		tree = shapely.strtree.STRtree(
			list(contours.values()))
		seps = collections.defaultdict(list)

		for sep in self._table_columns:
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

		concavity = self._options["region_concavity"]
		if concavity != "bounds":
			try:
				concavity = float(concavity)
			except:
				raise click.BadOptionUsage(
					"region-concavity", "must be numeric or 'bounds'")
		self._concavity = concavity

		self._seq_merge = dict([
			(("regions", "TABULAR"), SequentialMerger(
				cohesion=(0.5, 0.9), max_error=0.05, line_limit=None)),
			(("regions", "TEXT"), SequentialMerger(
				cohesion=(0.6, 0.8), max_error=0.01, line_limit=3))
		])

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".dewarped.contours.zip").exists() and (
				self._overwrite or (
					not p.with_suffix(".aggregate.contours.zip").exists())

	def _compute_order(self, page, polygons):
		mag = page.magnitude(dewarped=True)
		fringe = self._options["fringe"] * mag
		return polygon_order(polygons, fringe=fringe)

	def concavity(self, page, polygon):
		concavity = self._concavity

		if concavity == "bounds":
			return shapely.geometry.box(*polygon.bounds)
		elif concavity > 1:
			from origami.concaveman import concaveman2d

			mag = page.magnitude(dewarped=True)
			ext = np.array(polygon.exterior.coords)
			pts = concaveman2d(
				ext,
				scipy.spatial.ConvexHull(ext).vertices,
				concavity=concavity,
				lengthThreshold=mag * self._options["region_detail"])
			return shapely.geometry.Polygon(pts)
		else:  # disable concaveman
			return polygon

	def union(self, page, shapes):
		return self.concavity(page, shapely.ops.cascaded_union(shapes))

	def aggregate_by_predictor(self, page, contours, line_counts):
		# modify convexity.
		contours = [(k, self.concavity(page, polygon)) for k, polygon in contours]

		# names.
		for k, contour in contours:
			contour.name = "/".join(k)

		# group by predictor.
		by_predictor = collections.defaultdict(list)
		for k, contour in contours:
			by_predictor[k[:2]].append(contour)

		# aggregate.
		for k in list(by_predictor.keys()):
			by_predictor[k] = fixed_point(
				partial(
					aggregation,
					union=partial(self.union, page),
					threshold=self._options["maximum_overlap"],
					combine=line_counts.combine),
				by_predictor[k],
				lambda x: set([p.wkt for p in x]))

		# export.
		for k, contours in by_predictor.items():
			for contour in contours:
				yield tuple(contour.name.split("/")), contour

	def xycut_orders(self, page, aggregate):
		blocks_by_class = collections.defaultdict(list)
		for block_path, block in aggregate:
			blocks_by_class[tuple(block_path[:2])].append((block_path, block))
		blocks_by_class[("*", )] = aggregate

		return dict(
			(block_class, self._compute_order(page, v))
			for block_class, v in blocks_by_class.items())

	def sequential_merge(self, page, orders, noaggregate, aggregate, line_counts):
		aggregate = dict(aggregate)
		new_aggregate = dict()

		for path, regions in orders.items():
			if path[0] == "*":
				continue

			merge = self._seq_merge.get(path[:2])
			if merge is None:
				for r in regions:
					new_aggregate[r] = aggregate[r]
				continue

			error_overlap = Overlap(
				dict(noaggregate),
				set(aggregate.keys()) - set([path[:2]]))

			for name, shape in merge(
				regions,
				[aggregate[path] for path in regions],
				partial(self.union, page),
				error_overlap,
				line_counts):

				new_aggregate[name] = shape

		return new_aggregate.items()

	def process(self, page_path: Path):
		blocks = self.read_dewarped_blocks(page_path)

		separators = Separators(
			page_path,
			self.read_dewarped_separators(page_path))

		if not blocks:
			return

		warped_blocks = self.read_blocks(page_path)
		warped_lines = self.read_lines(page_path, warped_blocks)
		line_counts = LineCounts(warped_lines)

		page = list(blocks.values())[0].page

		zf_path = page_path.with_suffix(".dewarped.contours.zip")
		with zipfile.ZipFile(zf_path, "r") as zf:
			meta = zf.read("meta.json")

		noaggregate = [(k, block.image_space_polygon) for k, block in blocks.items()]

		aggregate = list(self.aggregate_by_predictor(page, noaggregate, line_counts))
		orders = self.xycut_orders(page, aggregate)

		aggregate = self.sequential_merge(
			page, orders, noaggregate, aggregate, line_counts)

		table_data = separators.detect_table_data(dict(aggregate))
		with atomic_write(
			page_path.with_suffix(".tables.json"),
			mode="wb", overwrite=self._overwrite) as f:
			f.write(json.dumps(table_data).encode("utf8"))

		zf_path = page_path.with_suffix(".aggregate.contours.zip")
		with atomic_write(zf_path, mode="wb", overwrite=self._overwrite) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				zf.writestr("meta.json", meta)
				for path, shape in aggregate:
					zf.writestr("/".join(path) + ".wkt", shape.wkt.encode("utf8"))

		if True:
			orders = self.xycut_orders(page, aggregate)

			orders = dict(("/".join(k), [
				"/".join(p) for p in ps]) for k, ps in orders.items())

			data = dict(
				version=1,
				order=orders)

			zf_path = page_path.with_suffix(".xycut.json")
			with atomic_write(zf_path, mode="w", overwrite=self._overwrite) as f:
				f.write(json.dumps(data))


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

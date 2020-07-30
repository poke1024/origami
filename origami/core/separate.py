import io
import numpy as np
import pickle
import json
import zipfile
import collections
import PIL.Image
import shapely.strtree
import intervaltree

from cached_property import cached_property

from origami.core.predict import PredictorType
from origami.core.math import inset_bounds, outset_bounds


class Separators:
	def __init__(self, segmentation, separators, widths):
		self._predictions = dict()
		for p in segmentation.predictions:
			if p.type == PredictorType.SEPARATOR:
				self._predictions[p.name] = p

		parsed_seps = collections.defaultdict(list)
		all_seps = []
		for k, geom in separators.items():
			prediction_name, prediction_type = k[:2]
			prediction = self._predictions[prediction_name]
			parsed_seps[prediction.classes[prediction_type]].append(geom)
			geom.name = "/".join(k)
			all_seps.append(geom)

		self._by_path = separators
		self._all_seps = all_seps
		self._parsed_seps = parsed_seps
		self._widths = widths  # sep width on warped page

	@property
	def by_path(self):
		return self._by_path

	@property
	def geoms(self):
		return self._all_seps

	@cached_property
	def _tree(self):
		return shapely.strtree.STRtree(self._all_seps)

	def query(self, shape):
		return self._tree.query(shape)

	def label(self, name):
		prediction_name, prediction_label = name.split("/")[:2]
		return self._predictions[prediction_name].classes[prediction_label]

	def for_label(self, name):
		return self._parsed_seps[self.label(name)]

	def check_obstacles(self, bounds, obstacles, fringe=0):
		bounds = inset_bounds(bounds, fringe)
		obstacles = set([self.label(o) for o in obstacles])
		box = shapely.geometry.box(*bounds)
		for sep in self.query(box):
			if self.label(sep.name) in obstacles:
				if box.intersects(sep):
					return True
		return False

	def width(self, name):
		return self._widths.get(tuple(name.split("/")), 1)


def extract_segments(geom):
	geom_type = geom.geom_type
	if geom_type == "LineString":
		return [geom]
	elif geom_type == "MultiLineString":
		return geom.geoms
	elif geom_type in ("Point", "MultiPoint"):
		return []
	elif geom_type == "GeometryCollection":
		result = []
		for g in geom.geoms:
			result.extend(extract_segments(g))
		return result
	else:
		raise RuntimeError(
			"unexpected geom type %s" % geom_type)


class ObstacleSampler:
	def __init__(self, separators, thickness_delta=None):
		self._separators = separators
		self._label = separators.label
		self._thickness_delta = thickness_delta

		self._direction = {
			self._label("separators/H"): 0,
			self._label("separators/V"): 1,
			self._label("separators/T"): 1
		}

	def __call__(self, gap):
		if gap.du < 0.5 or gap.dv < 0.5:
			return 0

		k = 5
		box = shapely.geometry.box(*outset_bounds(gap.bounds, k))

		flow = intervaltree.IntervalTree()
		obst = intervaltree.IntervalTree()

		flow_widths = []
		flow_width_weights = []

		for sep in self._separators.query(box):
			intersection = sep.intersection(box)
			if intersection is None or intersection.is_empty:
				continue

			label = self._label(sep.name)
			sep_dir = self._direction[label]

			for segment in extract_segments(intersection):
				minx, miny, maxx, maxy = segment.bounds
				smin = (minx, miny)
				smax = (maxx, maxy)

				if sep_dir == gap.axis:
					uax = gap.axis
					obst.addi(smin[uax], smax[uax] + 1, True)
				else:
					vax = 1 - gap.axis
					flow.addi(smin[vax], smax[vax] + 1, True)

					flow_widths.append(self._separators.width(sep.name))
					flow_width_weights.append(smax[vax] - smin[vax])

		flow.merge_overlaps(strict=False)
		obst.merge_overlaps(strict=False)

		flow_score = sum(i.length() for i in flow) / gap.dv
		obst_score = sum(i.length() for i in obst) / gap.du

		if self._thickness_delta and flow_widths:
			w = np.average(flow_widths, weights=flow_width_weights)
			delta_t = self._thickness_delta(w)
			obst_score -= delta_t
			flow_score += delta_t

		score = gap.du * gap.dv  # i.e. largest whitespace area
		score = (score * (1 - obst_score)) * (1 + flow_score)

		return score

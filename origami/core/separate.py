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
	def __init__(self, segmentation, separators):
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
			geom.name = "/".join(k[:2])
			all_seps.append(geom)

		self._all_seps = all_seps
		self._parsed_seps = parsed_seps

	@cached_property
	def _tree(self):
		return shapely.strtree.STRtree(self._all_seps)

	def query(self, shape):
		return self._tree.query(shape)

	def label(self, name):
		prediction_name, prediction_label = name.split("/")
		return self._predictions[prediction_name].classes[prediction_label]

	def for_label(self, name):
		return self._parsed_seps[self.label(name)]

	def check_obstacles(self, bounds, obstacles, fringe=0):
		bounds = inset_bounds(bounds, fringe)
		obstacles = set([self.label(o) for o in obstacles])
		box = shapely.geometry.box(*bounds)
		for sep in self.query(box):
			if box.intersects(sep):
				if self.label(sep.name) in obstacles:
					return True
		return False


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
	def __init__(self, separators):
		self._separators = separators
		self._label = separators.label

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

		flow.merge_overlaps(strict=False)
		obst.merge_overlaps(strict=False)

		flow_score = sum(i.length() for i in flow) / gap.dv
		obst_score = sum(i.length() for i in obst) / gap.du

		score = gap.dv
		score = (score * (1 - obst_score)) * (1 + flow_score)

		return score

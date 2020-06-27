import io
import numpy as np
import pickle
import json
import zipfile
import collections
import PIL.Image
import shapely.strtree

from cached_property import cached_property

from origami.core.page import Page, Annotations
from origami.core.math import inset_bounds
from origami.core.predict import PredictorType


Predictor = collections.namedtuple(
	"Predictor", ["type", "name", "classes"])


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


class Segmentation:
	def __init__(self, predictions):
		self._predictions = tuple(predictions)

	@property
	def predictions(self):
		return self._predictions

	@property
	def size(self):
		shape = self._predictions[0].labels.shape
		return tuple(reversed(list(shape)[:2]))

	@staticmethod
	def _read_pickle(f):
		from origami.core.predict import Prediction

		data = pickle.load(f)

		return Segmentation(
			[Prediction.deserialize(v) for k, v in data])

	@staticmethod
	def open_pickle(path):
		with open(path, "rb") as f:
			return Segmentation._read_pickle(f)

	@staticmethod
	def open(path):
		from origami.core.predict import Prediction, PredictorType

		predictions = []
		with zipfile.ZipFile(path, "r") as zf:
			tasks = []
			for name in zf.namelist():
				if name.endswith(".png"):
					stem = name.rsplit('.', 1)[0]
					tasks.append(stem)

			for task in tasks:
				with io.BytesIO(zf.read(task + ".png")) as f:
					im = PIL.Image.open(f)
					im.load()
				meta = json.loads(zf.read(task + ".json"))

				t = PredictorType[meta["type"]]
				classes = t.classes(meta["classes"])
				predictions.append(Prediction(
					t, meta["name"], np.array(im), classes))

		return Segmentation(predictions)

	def save(self, path):
		with zipfile.ZipFile(path, "w") as zf:
			for p in self._predictions:
				with io.BytesIO() as f:
					p.colorized.save(f, "png", optimize=True)
					zf.writestr("%s.png" % p.name, f.getvalue())

				meta = dict(
					type=p.type.name,
					name=p.name,
					classes=dict([(m.name, m.value) for m in p.classes]))
				zf.writestr("%s.json" % p.name, json.dumps(meta))

	@staticmethod
	def read_predictors(path):
		predictors = []
		with zipfile.ZipFile(path, "r") as zf:
			for name in zf.namelist():
				if name.endswith(".json"):
					data = json.loads(zf.read(name))
					predictors.append(Predictor(**data))
		return predictors


class SegmentationPredictor:
	def __init__(self, models_path, target="quality"):
		import origami.core.predict as predict

		if target == "speed":
			loaded = predict.load([
				(predict.NetPredictor, "v3/sep/1"),
				(predict.NetPredictor, "v3/blkx/2"),
			], models_path=models_path)

			self._predictors = [
				loaded["v3/sep/1"],
				loaded["v3/blkx/2"]
			]
		elif target == "quality":
			loaded = predict.load([
				(predict.NetPredictor, "v3/sep/1"),
				(predict.NetPredictor, "v3/sep/2"),
				(predict.NetPredictor, "v3/sep/4"),
				(predict.NetPredictor, "v3/blkx/1"),
				(predict.NetPredictor, "v3/blkx/2"),
				(predict.NetPredictor, "v3/blkx/3"),
				(predict.NetPredictor, "v3/blkx/4"),
				(predict.NetPredictor, "v3/blkx/5"),
				], models_path=models_path)

			self._predictors = [
				predict.VotingPredictor(
					loaded["v3/sep/1"],
					loaded["v3/sep/2"],
					loaded["v3/sep/4"],
					name="separators"),
				predict.VotingPredictor(
					loaded["v3/blkx/1"],
					loaded["v3/blkx/2"],
					loaded["v3/blkx/3"],
					loaded["v3/blkx/4"],
					loaded["v3/blkx/5"],
					name="regions")]
		else:
			raise ValueError(target)

	def __call__(self, path):
		page = Page(path)
		return Segmentation([p(page) for p in self._predictors])


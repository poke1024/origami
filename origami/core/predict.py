import numpy as np
import matplotlib.pyplot as plt

import json
import cv2
import math
import re
import io
import itertools
import enum
import PIL.Image
import logging

from pathlib import Path
from tqdm import tqdm

from origami.core.math import Orientation


def category_colors(n):
	colors = plt.get_cmap("tab10").colors
	return np.array(list(colors)).flatten() * 255


def colorize(labels, background=None):
	n_labels = np.max(labels) + 1
	colors = category_colors(n_labels)

	if background:
		colors[background * 3 + 0] = 255
		colors[background * 3 + 1] = 255
		colors[background * 3 + 2] = 255

	im = PIL.Image.fromarray(labels, "P")
	pil_pal = np.zeros((768,), dtype=np.uint8)
	pil_pal[:len(colors)] = colors
	im.putpalette(pil_pal)

	return im


class Tile:
	def __init__(self, outer, inner):
		self._outer = outer
		self._inner = inner

	@property
	def outer(self):
		return self._outer

	@property
	def inner(self):
		return self._inner

	def read_outer(self, pixels):
		x0, y0, x1, y1 = self._outer
		return pixels[y0:y1, x0:x1]

	def write_inner(self, labels, data):
		x0, y0, x1, y1 = self._inner
		dx, dy = np.array(self._inner[:2]) - np.array(self._outer[:2])
		labels[y0:y1, x0:x1] = data[dy:dy + (y1 - y0), dx:dx + (x1 - x0)]


class Tiles:
	def __init__(self, tile_size, beta0=50):
		self._tile_size = tile_size
		assert all(beta0 < s for s in tile_size)
		self._beta0 = beta0

	def _tiles_1(self, full_size, tile_size):
		if tile_size == full_size:
			yield (0, full_size), (0, full_size)
		else:
			n_steps = math.ceil(full_size / tile_size)

			while True:
				r = (full_size - tile_size) / ((n_steps - 1) * tile_size)

				if tile_size * (1 - r) > self._beta0:
					break

				n_steps += 1

			x0 = []
			x1 = []
			for i in range(n_steps):
				x = round(i * tile_size * r)
				x -= max(0, x + tile_size - full_size)

				x0.append(x)
				x1.append(x + tile_size)

			for i in range(n_steps):
				if i > 0:
					x0_inner = (x1[i - 1] + x0[i]) // 2
				else:
					x0_inner = 0

				if i < n_steps - 1:
					x1_inner = (x1[i] + x0[i + 1]) // 2
				else:
					x1_inner = full_size

				yield (x0[i], x1[i]), (x0_inner, x1_inner)

	def __call__(self, full_size):
		p = [list(self._tiles_1(f, t)) for f, t in zip(full_size, self._tile_size)]
		for x, y in itertools.product(*p):
			(x0, x1), (xi0, xi1) = x
			(y0, y1), (yi0, yi1) = y
			yield Tile((x0, y0, x1, y1), (xi0, yi0, xi1, yi1))


def load(what, **kwargs):
	loaded = dict()
	for c, name in tqdm(what, desc="loading models"):
		loaded[name] = c(name, **kwargs)
	return loaded


class RegionLabel(enum.Enum):
	pass


class SeparatorLabel(enum.Enum):
	@property
	def orientation(self):
		if self.name.startswith("H"):
			return Orientation.H
		else:
			return Orientation.V


class PredictorType(enum.Enum):
	REGION = 1
	SEPARATOR = 2

	def classes(self, c):
		if self == PredictorType.REGION:
			return enum.Enum("RegionLabel", c, type=RegionLabel)
		elif self == PredictorType.SEPARATOR:
			return enum.Enum("SeparatorLabel", c, type=SeparatorLabel)
		else:
			raise ValueError(self)


def _check_predictor_name(name):
	if not re.match(r"^[A-Za-z0-9/]+$", name):
		raise ValueError("illegal predictor name '%s'" % name)


class Predictor:
	pass


class NetPredictor(Predictor):
	def __init__(self, model_name, models_path, name=None):
		if not re.match(r"^[a-z0-9/]+$", model_name):
			raise ValueError("illegal model name '%s'" % model_name)

		if name is None:
			name = model_name
		self._name = name
		_check_predictor_name(self._name)

		models_path = Path(models_path)
		network_path = models_path / model_name

		for filename in ("meta.json", "model.h5"):
			asset_path = network_path / filename
			if not asset_path.exists():
				raise FileNotFoundError("no model file found at %s" % asset_path)

		with open(network_path / "meta.json", "r") as f:
			meta = json.loads(f.read())
		classes = meta["classes"]

		import segmentation_models

		# the following commented code fails to work.
		# see https://github.com/qubvel/segmentation_models/issues/153
		# and https://stackoverflow.com/questions/54835331/how-do-i-load-a-keras-saved-model-with-custom-optimizer
		'''
		model = getattr(segmentation_models, meta["model"])(
			meta["backbone"],
			classes=len(classes),
			activation="softmax")
		logging.info("loading model at %s" % str(network_path / "model.h5"))
		model.load_weights(str(network_path / "model.h5"))
		'''

		# note that we need keras.models.load_model, as
		# tensorflow.keras.models.load_model won't work,
		# see https://github.com/keras-team/keras-contrib/issues/488
		# and https://github.com/tensorflow/tensorflow/issues/25200
		# so we need a separate keras installation here.
		from keras.models import load_model

		model = load_model(str(network_path / "model.h5"), compile=False)
			
		self._preprocess = segmentation_models.get_preprocessing(
			meta["backbone"])

		self._model = model
		self._full_size = tuple(meta["full_size"])
		self._full_shape = tuple(reversed(self._full_size))
		self._tile_size = tuple(meta["tile_size"])

		self._tiles = list(Tiles(
			self._tile_size,
			beta0=meta["tile_beta"])(meta["full_size"]))

		self._type = PredictorType[meta["type"].upper()]
		self._classes = self._type.classes(dict((v, i) for i, v in enumerate(classes)))

	@property
	def type(self):
		return self._type

	@property
	def name(self):
		return self._name

	@property
	def classes(self):
		return self._classes

	@property
	def size(self):
		return self._full_size

	@property
	def tile_size(self):
		return self._tile_size

	def _predict(self, page, labels=None, verbose=False):
		if labels is None:
			im = page.image.convert("RGB")
			pixels = np.array(im)

			net_input = cv2.resize(pixels, self._full_size, interpolation=cv2.INTER_AREA)
			net_input = cv2.cvtColor(net_input, cv2.COLOR_BGR2RGB)
			net_input = self._preprocess(net_input)

			labels = np.empty(self._full_shape, dtype=np.uint8)

			if verbose:
				tiles = tqdm(self._tiles, desc="prediction")
			else:
				tiles = self._tiles

			for tile in tiles:
				tile_pixels = tile.read_outer(net_input)
				tile_pixels = np.expand_dims(tile_pixels, axis=0)
				pr_mask = self._model.predict(tile_pixels)
				tile_labels = np.argmax(pr_mask.squeeze(), axis=-1).astype(np.uint8)
				tile.write_inner(labels, tile_labels)

		return Prediction(
			self.type, self.name,
			labels,
			self._classes)

	@property
	def background(self):
		return self._classes["BACKGROUND"]

	def __call__(self, page):
		return self._predict(page)


def _majority_vote(data, undecided=0):
	data = np.array(data, dtype=data[0].dtype)
	n_labels = np.max(data) + 1

	counts = np.zeros(
		(n_labels,) + data[0].shape, dtype=np.int32)
	for label in range(n_labels):
		for pr in data:
			counts[label][pr == label] += 1

	counts = np.dstack(counts)
	most_freq = np.argmax(counts, axis=-1).astype(data.dtype)

	order = np.argsort(counts)
	candidates_count = np.take_along_axis(counts, order[:, :, -2:], axis=-1)

	if candidates_count.shape[-1] >= 2:
		tie = np.logical_not(candidates_count[:, :, 0] < candidates_count[:, :, 1])
		most_freq[tie] = undecided

	return most_freq


class VotingPredictor(Predictor):
	def __init__(self, *predictors, name=None):
		if not all(p.type == predictors[0].type for p in predictors):
			raise ValueError("predictor need to have same predictor types")
		self._predictors = predictors
		self._undecided = predictors[0].background.value

		if name is None:
			name = "&".join([p.name for p in predictors])
		self._name = name
		_check_predictor_name(self._name)

	def __call__(self, pixels):
		predictions = [p(pixels) for p in self._predictors]
		return Prediction(
			self.type, self.name,
			_majority_vote([p.labels for p in predictions], self._undecided),
			self._predictors[0].classes)

	@property
	def name(self):
		return self._name

	@property
	def type(self):
		return self._predictors[0].type


class Prediction:
	def __init__(self, type, name, labels, classes):
		self._type = type
		self._name = name
		self._labels = labels
		self._classes = classes
		self._background = self._classes["BACKGROUND"]

	@property
	def type(self):
		return self._type

	@property
	def name(self):
		return self._name

	@property
	def background_label(self):
		return self._background

	@property
	def labels(self):
		return self._labels

	@property
	def classes(self):
		return self._classes

	@property
	def colorized(self):
		return colorize(self._labels, self.background_label.value)

	@staticmethod
	def deserialize(data):
		enum_name, enum_dict, labels_data = data
		if enum_name == "RegionLabel":
			t = PredictorType.REGION
		elif enum_name == "SeparatorLabel":
			t = PredictorType.SEPARATOR
		else:
			raise ValueError(enum_name)

		classes = t.classes(enum_dict)

		with io.BytesIO(labels_data) as f:
			np_data = np.load(f)
			labels = np_data["arr_0"]

		return Prediction(t, t.name.lower() + "s", labels, classes)

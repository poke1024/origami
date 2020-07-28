import PIL
import PIL.ImageOps
import png

import math
import numpy as np
import time
import collections
import logging
import os
import json
import sys
import itertools
import skimage.filters

from tqdm import tqdm
from pathlib import Path

from. gen import annotations
from .gen import labels
from .gen import binarize
from .gen import warp


IMG_INTERPOLATION = PIL.Image.LANCZOS
MIN_TILE_OVERLAP = 50


class WarpAugmentation:
	def __init__(self, label_set):
		self.kind = "warping"
		self._label_set = label_set

	def name(self, n):
		return "WARP-" + n

	def process(self, psd_path, ground_truth):
		return warp.warp_images(
			ground_truth, self._label_set, name=psd_path.stem)


class Target:
	def __init__(self, path, codes, size, tile_size):
		self.path = path
		self.codes = codes

		self.size = size
		self.tile_size = tile_size

		# label index mapping.
		lut = np.zeros((256,), dtype=np.uint8)
		for i, labels in enumerate(codes.values()):
			for label in labels:
				lut[label.index] = i
		self.lut = lut


class Generator:
	def __init__(self, target):
		self._target = target

	def _tiles(self, path):
		t = list(tiles(self._target.size, self._target.tile_size))
		for (x0, x1), (y0, y1) in t:
			if len(t) == 1:
				yield path, (x0, x1), (y0, y1)
			else:
				yield path.parent / (path.stem + ("-T-%d-%d" % (x0, y0)) + path.suffix), (x0, x1), (y0, y1)

	def _paths(self, path):
		for p, _, _ in self._tiles(path):
			yield p

	def _write_tiles(self, path, pixels, palette=None):
		assert tuple(pixels.shape) == tuple(reversed(self._target.size))

		for tile_path, (x0, x1), (y0, y1) in self._tiles(path):
			tile_pixels = pixels[y0:y1, x0:x1]

			if palette:
				with open(tile_path, "wb") as f:
					w = png.Writer(x1 - x0, y1 - y0, palette=palette, bitdepth=8)
					w.write(f, tile_pixels)
			else:
				PIL.Image.fromarray(tile_pixels).save(tile_path)

	def _write_annotations(self, path, annotations, palette):
		self._write_tiles(path, np.vstack(annotations), palette)


class TrainingImageGenerator(Generator):
	def __init__(self, target, name):
		super().__init__(target)
		basepath = target.path
		(basepath / "images").mkdir(exist_ok=True)
		self._path = basepath / "images" / (name + ".png")

	@property
	def kind(self):
		return "images"

	@property
	def stats(self):
		return None

	@property
	def paths(self):
		return list(self._paths(self._path))

	def write(self, ground_truth):
		im = PIL.Image.fromarray(ground_truth.unbinarized, "L")
		small_image = im.resize(self._target.size, IMG_INTERPOLATION)
		self._write_tiles(self._path, np.array(small_image))


class LabelsGenerator(Generator):
	def __init__(self, kind, target, name):
		super().__init__(target)
		self._kind = kind
		basepath = target.path
		(basepath / kind).mkdir(exist_ok=True)
		self._p_path = basepath / kind / (name + "_P.png")
		self._c_path = basepath / kind / (name + "_C.png")

	@property
	def kind(self):
		return self._kind

	@property
	def paths(self):
		return list(self._paths(self._p_path)) + list(self._paths(self._c_path))

	def _write_labels(self, labels):
		# fastai ignores the palette index, and instead want the gray value
		# to be the label :-(

		self._write_annotations(
			self._p_path,
			labels,
			[[i, i, i] for i in range(len(self._target.codes))])

		# like _write_c_labels, but with a correct palette, for visual control.
		self._write_annotations(
			self._c_path,
			labels,
			[labels[0].color for labels in self._target.codes.values()])

	@property
	def stats(self):
		counts = []

		w, h = self._target.tile_size
		n_pixels = w * h

		for p, _, _ in self._tiles(self._p_path):
			labels = np.array(PIL.Image.open(p))

			f = np.bincount(
				labels.flatten(),
				minlength=len(self._target.codes))
			assert np.sum(f) == n_pixels

			counts.append(f)

		return np.mean(np.array(counts), axis=0)


class PixelLabelGenerator(LabelsGenerator):
	def __init__(self, target, name):
		super().__init__("pixels", target, name)

	def write(self, ground_truth):
		annotations = ground_truth.annotations(
			"master", shape=tuple(reversed(self._target.size)))

		annotations = annotations.apply_lut(self._target.lut)

		self._write_labels(annotations.labels)


class RegionLabelGenerator(LabelsGenerator):
	def __init__(self, target, name):
		super().__init__("regions", target, name)

	def write(self, ground_truth):
		if not ground_truth.has_annotations("regions"):
			raise RuntimeError(
				"no regions annotations found for %s. did you run gen_seg.py?" % ground_truth.path)

		annotations = ground_truth.annotations(
			"regions", shape=tuple(reversed(self._target.size)))

		annotations = annotations.apply_lut(self._target.lut)

		self._write_labels(annotations.labels)


def tiles_1(full_size, tile_size):
	if tile_size == full_size:
		yield 0, full_size
	else:
		n_steps = 2

		while True:
			overlap = int(((n_steps * tile_size) - full_size) / (n_steps - 1))
			if overlap >= MIN_TILE_OVERLAP:
				break
			n_steps += 1

		for i in range(0, n_steps):
			y = i * (tile_size - overlap)
			y -= max(0, y + tile_size - full_size)
			yield y, y + tile_size


def tiles(full_size, tile_size):
	p = [tiles_1(f, t) for f, t in zip(full_size, tile_size)]
	return itertools.product(*p)


class Converter:
	def __init__(self, out_path, out_codes, image_size, tile_size, label_set):
		# note: our approach here is to resize everything to (small_w, small_h)
		# and blatantly ignore aspect ratio. this means images might get (slightly)
		# stretched but we can take full advantage of the available pixels for
		# training our NN.

		self._augmentations = (None, WarpAugmentation(label_set))

		self._label_set = label_set

		# outgoing parameters.
		self._out_codes = out_codes
		self._out_path = out_path

		# statistics for computing class loss weights.
		self._stats = collections.defaultdict(list)
		self._limit = None

		# with open(out_path / "valid.txt", "r") as f:
		# self._valid = set(s.strip() for s in f.readlines())

		self._logger = logging.getLogger('converter')

		self._target = Target(
			self._out_path, out_codes, image_size, tile_size)

	def __call__(self, gt_ref):
		for augmentation in self._augmentations:
			self._psd(
				gt_ref,
				augmentation)

	def _generators(self, psd_path, augmentation):
		if augmentation:
			name = augmentation.name(psd_path.stem)
		else:
			name = psd_path.stem

		return (
			TrainingImageGenerator(self._target, name),
			PixelLabelGenerator(self._target, name),
			RegionLabelGenerator(self._target, name)
		)

	def _psd(self, gt_ref, augmentation=None):
		psd_path = gt_ref.annotated_path

		self._logger.info(
			"processing [%s] %s." % (augmentation.kind if augmentation else "default", psd_path))

		generators = self._generators(psd_path, augmentation)

		generated_paths = list(itertools.chain(*[g.paths for g in generators]))

		if all(p.is_file() for p in generated_paths):  # skip?
			for g in generators:
				stats = g.stats
				if stats is not None:
					self._stats[g.kind].append(stats)
			return

		# if self._limit and len(self._stats) >= self._limit:
		#    return  # for debugging

		ground_truth = gt_ref.load(self._logger)

		if augmentation:
			ground_truth = augmentation.process(psd_path, ground_truth)

		# generate artefacts.
		for g in generators:
			g.write(ground_truth)

			if augmentation is None:  # only count 1 non-augmented version.
				stats = g.stats
				if stats is not None:
					self._stats[g.kind].append(stats)

	def weights(self):
		w, h = self._target.tile_size
		n_pixels = w * h

		weights = dict()
		for k, freq in self._stats.items():
			n_classes = len(self._out_codes)

			freq = np.array(freq)

			w = np.zeros(shape=(n_classes,), dtype=np.float32)
			for i in range(n_classes):
				f = freq[:, i]
				if (f > 0).any():
					f = f[f > 0]
					w[i] = n_pixels / np.median(f)
			weights[k] = w.tolist()
		return weights


class Preprocessor:
	def __init__(self, data_path):
		self._data_path = Path(data_path)

		if not (data_path / "preprocessed").exists():
			(data_path / "preprocessed").mkdir()

		self._label_set = None

	def _gen_valid(self, inputs, p=0.2):
		import random

		random.seed(1036536561063490465169)
		files = list(map(lambda x: x.document_path, inputs))
		random.shuffle(files)

		for i, valid in enumerate(np.array_split(range(len(files)), math.ceil(1 / p))):

			valid_path = Path(self._data_path / "preprocessed" / ("valid%d.txt" % (i + 1)))
			if valid_path.is_file():
				continue

			with open(valid_path, "w") as f:
				for j in valid:
					f.write(files[j].stem + ".png\n")

	def _gen_train(self, inputs, model_name, codes, image_size, tile_size):
		if image_size == tile_size:
			size_desc = "%dx%d" % image_size
		else:
			size_desc = "%dx%d_T%dx%d" % (*image_size, *tile_size)

		out_path = Path(self._data_path / ("preprocessed/%s_%s" % (model_name, size_desc)))
		out_path.mkdir(exist_ok=True)

		# actual processing.

		converter = Converter(
			out_path, codes, image_size, tile_size, self._label_set)
		# converter._limit = 5  # debugging

		for gt_ref in tqdm(inputs, desc="%dx%d" % image_size):
			converter(gt_ref)

		# code and weight files.

		import json

		with open(out_path / "codes.json", "w") as f:
			f.write(json.dumps(list(codes.keys())))

		weights = converter.weights()

		for kind, w in weights.items():
			with open(out_path / kind / "weights.json", "w") as f:
				f.write(json.dumps(w))

	def gen(self, json_path):
		corpus_path = Path(self._data_path / "corpus")

		with open(json_path, "r") as f:
			json_spec = json.loads(f.read())

		self._label_set = labels.LabelSet(json_spec)
		merge_spec = json_spec.get("postprocessing")

		inputs = annotations.collect_ground_truth(
			corpus_path, self._label_set, merge_spec)

		self._gen_valid(inputs)

		def create_training_data(full_size, tile_size=None):
			if tile_size is None:
				tile_size = full_size

			for codes_name, codes_mapping in self._label_set.codes.items():
				self._gen_train(inputs, codes_name, codes_mapping, full_size, tile_size)

		# specify which combination of tile and scaling to generate here.
		create_training_data((1280, 2400), (1280, 896))


if __name__ == "__main__":
	assert len(sys.argv) == 2

	data_path = Path(sys.argv[1])

	(data_path / "preprocessed").mkdir(exist_ok=True)

	logging.basicConfig(
		level=logging.ERROR,
		format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
		datefmt='%m-%d %H:%M',
		filename=data_path / "preprocessed/build.log",
		filemode='w')

	binarize.gen_binarized(data_path)

	p = Preprocessor(data_path)

	script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
	p.gen(script_dir / "custom" / "bbz.json")

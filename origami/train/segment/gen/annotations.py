import numpy as np
import PIL.Image
import itertools
import cv2
import importlib

from psd_tools import PSDImage
from psd_tools.constants import BlendMode

from .merger import SegmentMerger
from .transform import Resize
from .regions import AnnotationsGenerator


class Annotations:
	def __init__(self, label_set, labels, img_path=None):
		self._label_set = label_set
		self._labels = labels  # 2d array
		self._img_path = img_path
		self._skew = None

	@property
	def shape(self):
		return self._labels.shape

	@property
	def palette(self):
		return self._label_set.palette

	@property
	def image(self):
		im = PIL.Image.fromarray(self._labels, "P")
		im.putpalette(self.palette)
		return im

	@property
	def labels(self):
		return self._labels

	@property
	def mutable_labels(self):
		return self._labels

	def apply_lut(self, lut):
		return Annotations(
			self._label_set, lut[self._labels], self._img_path)

	def mask(self, *labels):
		n_labels = self._label_set.n_labels
		lut = np.zeros((n_labels, ), dtype=np.bool)
		for label in labels:
			lut[label.index] = True
		return lut[self._labels]

	def mask_by_name(self, *names):
		return self.mask(*[
			self._label_set.label_from_name(name) for name in names])

	def _selected_labels(self, masked_labels):
		h, w = self.shape
		labels = np.empty((h, w), dtype=np.uint8)
		labels.fill(self._label_set.background.index)

		m = self.mask(*masked_labels)
		labels[m] = self._labels[m]
		return labels

	@property
	def separator_labels(self):
		return self._selected_labels([
			label for label in self._label_set.labels
			if label.is_separator])

	@property
	def non_separator_labels(self):
		return self._selected_labels([
			label for label in self._label_set.labels
			if not label.is_separator])

	def _find_segment_components(self, *labels):
		mask = self.mask(*labels)
		n, c = cv2.connectedComponents(mask.astype(np.uint8))
		for i in range(1, n + 1):
			yield c == i

	@property
	def unprocessed_segments(self, min_pts=5):
		# detect segments based on label features.

		# min_pts of 5 works good to remove noise from annotations
		# that splitters seg lines. noise often has 3 or 4 pixels.

		from .segments import Segment

		sep_groups = [
			self._label_set.separators("h"),
			self._label_set.separators("v")
		]

		index = 1
		for mask in itertools.chain(*[
			self._find_segment_components(*x) for x in sep_groups]):

			if np.sum(mask.astype(np.uint8)) < min_pts:
				continue  # not enough points

			s = Segment.from_mask(
				self._label_set, self._labels, mask, str(index))

			if s is not None:
				yield s
				index += 1

	def merger(self, merge_spec):
		return SegmentMerger(
			merge_spec,
			self._label_set,
			self._labels,
			list(self.unprocessed_segments))

	def repaired_segments(self, merge_spec):
		return self.merger(merge_spec).segments

	def transform(self, t):
		return Annotations(self._label_set, t.labels(self._labels))


class GroundTruth:
	def __init__(self, ref, unbinarized, binarized, master):
		self._ref = ref

		assert binarized is not None
		assert master is not None

		self._binarized = binarized
		self._labels = dict(master=master)
		self._unbinarized = unbinarized

		images = [unbinarized, binarized, *self._labels.values()]
		assert all(im.shape[:2] == images[0].shape[:2] for im in images)

	def add_labels(self, name, labels):
		assert(labels.shape[:2] == self._unbinarized.shape[:2])
		self._labels[name] = labels

	def asset_path(self, *args, **kwargs):
		return self._ref.asset_path(*args, **kwargs)

	def transform(self, f):
		images = [self._unbinarized, self._binarized]
		images = list(map(lambda im: f("image", im), images))

		labels = dict((k, f("labels", l)) for k, l in self._labels.items())

		gt = GroundTruth(self._ref, images[0], images[1], labels["master"])
		for k, v in labels.items():
			if k != "master":
				gt.add_labels(k, v)
		return gt

	@property
	def path(self):
		return self._ref.path

	@property
	def shape(self):
		return self._unbinarized.shape

	@property
	def unbinarized(self):
		return self._unbinarized

	@property
	def binarized(self):
		return self._binarized

	@property
	def labels(self):
		return self._labels["master"]

	def _resize_bin(self, image, shape):
		if tuple(image.shape) == tuple(shape):
			return image

		resize = Resize(from_size=reversed(image.shape), to_size=reversed(shape))
		return resize.mask(image > 0).astype(np.uint8)

	def _resize_labels(self, image, shape):
		if tuple(image.shape) == tuple(shape):
			return image

		resize = Resize(from_size=reversed(image.shape), to_size=reversed(shape))
		return resize.labels(image, weights=self._ref.label_set.label_weights)

	def has_annotations(self, kind):
		return kind in self._labels

	def annotations(self, kind="master", shape=None, img_path=None):
		labels = self._labels[kind]

		if shape is None:
			shape = labels.shape

		# resize to prediction size.
		labels = self._resize_labels(labels, shape)

		if kind == "master":
			# base layer (binarization).

			# note: as soon as one pixel is marked in binarized as "interesting"
			# in the larger image, we want to have that pixel in the smaller
			# image as "interesting" as well. otherwise we would throw away
			# labels that might be very important (e.g. thin separators).

			binarized = self._resize_bin(self._binarized > 0, shape)  # white?

			# mask out non-binarized parts.
			labels[np.logical_not(binarized)] = self._ref.background.index

		return Annotations(self._ref.label_set, labels, img_path)


class Loader:
	def __init__(self, label_set, merge_spec):
		self._label_set = label_set
		self._merge_spec = merge_spec
		self._palette_image = PIL.Image.new('P', (16, 16))
		self._palette_image.putpalette(label_set.palette)
		self._background = label_set.background

	def _rgb2labels(self, pixels, bin_data=None, logger=None):
		ann_data = pixels.quantize(method=1, palette=self._palette_image)
		ann_data = np.array(ann_data, dtype=np.uint8)

		ann_rgb = PIL.Image.fromarray(ann_data, "P")
		ann_rgb.putpalette(self._palette_image.getpalette())
		ann_rgb = np.array(ann_rgb.convert("RGB", dither=PIL.Image.NONE))
		ignore = np.all(ann_rgb != np.array(pixels), axis=-1)

		if logger:
			if bin_data is not None:
				n_ignore = np.sum(np.logical_and(ignore, bin_data).astype(np.uint8))
			else:
				n_ignore = np.sum(ignore.astype(np.uint8))

			ignore_ratio = n_ignore / ann_rgb.size

			if n_ignore == 0:
				logger.info("no pixels ignored")
			elif ignore_ratio < 1 / 1000:
				logger.warning("ignored < 1%%%% pixels (%d)" % n_ignore)
			else:
				logger.warning("ignored > 1%%%% pixels (%d)" % n_ignore)

		if bin_data is not None:
			ignore = np.logical_or(ignore, np.logical_not(bin_data))

			# tmp.save(self._out_path / "images" / (psd_path.stem + ".debug.png"))
			# tmp.convert("RGB", dither=PIL.Image.NONE).save(self._out_path / "images" / (psd_path.stem + ".debug.png"))

			ann_data[ignore] = self._background.index

		return ann_data

	def _generate_regions(self, ground_truth):
		annotations = ground_truth.annotations()

		if self._merge_spec:
			try:
				segments = annotations.repaired_segments(self._merge_spec)

				gen = AnnotationsGenerator(
					self._label_set, self._merge_spec, annotations, segments)

				gen_module = importlib.import_module(self._merge_spec["generator"])
				ann = gen_module.generate(gen, ground_truth.unbinarized)
			except:
				print("error on generating data for ", ground_truth.path)
				raise
		else:
			ann = annotations  # don't do any postprocessing of annotations.

		im = ann.image

		im.save(ground_truth.asset_path(
			"seg", prefix="regions.", ext=".png"))

		debug_im = PIL.Image.blend(
			PIL.Image.fromarray(ground_truth.unbinarized).convert("RGB"),
			im.convert("RGB"),
			0.75)
		debug_im.save(ground_truth.asset_path(
			"seg", prefix="debug.regions.", ext=".jpg"), "JPEG", quality=75)

		return ann.labels

	def __call__(self, ground_truth_ref, psd_path, img_path, logger=None, generate=True):
		bin_data = None
		ann_data = None
		image_size = 0

		unbinarized = np.array(PIL.Image.open(img_path).convert('L'))

		psd = PSDImage.open(str(psd_path))

		for layer in psd:
			if layer.blend_mode == BlendMode.NORMAL:
				# assert layer.offset == (0, 0)

				layer_image = layer.topil().convert('L')
				image_size = layer.size
				bin_data = np.array(layer_image)

			elif layer.blend_mode == BlendMode.MULTIPLY:
				layer_image = layer.topil()

				alpha_mask = PIL.Image.fromarray((np.array(layer_image)[:, :, 3] > 128).astype(np.uint8) * 255)
				layer_image = layer_image.convert("RGB")  # remove alpha channel

				ann_rgb_image = PIL.Image.new("RGB", image_size, (255, 255, 255))
				ann_rgb_image.paste(layer_image, layer.offset, alpha_mask)

				# now reduce to palette
				# tmp = rgb_image.im.convert("P", 0, self._palette_image.im)
				# tmp = rgb_image._new(tmp)

				ann_data = self._rgb2labels(ann_rgb_image, bin_data, logger)

		gt = GroundTruth(ground_truth_ref, unbinarized, bin_data, ann_data)

		asset_path = gt.asset_path("seg", prefix="regions.", ext=".png")
		if asset_path.is_file():
			gt.add_labels("regions", np.array(PIL.Image.open(asset_path)))
		else:
			gt.add_labels("regions", self._generate_regions(gt))

		return gt


class GroundTruthRef:
	def __init__(self, psd_path, image_ext, label_set, merge_spec):
		self._path = psd_path
		self._image_ext = image_ext
		self._label_set = label_set
		self._loader = Loader(label_set, merge_spec)

	@property
	def path(self):
		return self._path

	def load(self, logger=None):
		return self._loader(
			self, self.annotated_path, self.document_path, logger)

	@property
	def annotated_path(self):
		return self._path

	@property
	def document_path(self):
		return self.asset_path("img", ext=self._image_ext)

	@property
	def label_set(self):
		return self._label_set

	@property
	def background(self):
		return self._label_set.background

	def asset_path(self, kind, prefix="", ext=".png"):
		container = self._path.parent.parent / kind
		container.mkdir(exist_ok=True)
		return container / (prefix + self._path.stem + ext)


def collect_ground_truth(corpus_path, label_set, merge_spec):
	def iter_int_dir(p):
		return filter(lambda d: d.stem.isdigit(), p.iterdir())

	def gather_files(ann_path):
		for ann_file in ann_path.iterdir():
			if ann_file.is_file() and ann_file.suffix == ".psd":
				yield ann_file

	# scanning.

	inputs = []

	for year_path in iter_int_dir(corpus_path):
		for month_path in iter_int_dir(year_path):
			for day_path in iter_int_dir(month_path):
				ann_path = day_path / "ann"
				if ann_path.is_dir():
					for ann_file in gather_files(ann_path):
						inputs.append(GroundTruthRef(
							ann_file, ".png", label_set, merge_spec))

	ann_path = corpus_path / "0000" / "ann"
	if ann_path.is_dir():
		for ann_file in gather_files(ann_path):
			inputs.append(GroundTruthRef(
				ann_file, ".jpg", label_set, merge_spec))

	return inputs

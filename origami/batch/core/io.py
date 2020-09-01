#!/usr/bin/env python3

import enum
import json
import zipfile
import collections
import shapely.wkt
import click

from pathlib import Path
from contextlib import contextmanager
from cached_property import cached_property
from atomicwrites import atomic_write

from origami.core.page import Page
from origami.core.predict import PredictorType
from origami.core.block import Block, Line
from origami.core.separate import Separators


def find_data_path(page_path):
	return page_path.with_suffix(".out")


class Stage(enum.Enum):
	WARPED = 0
	DEWARPED = 1
	AGGREGATE = 2
	RELIABLE = 3
	ANY = -1

	@property
	def is_dewarped(self):
		return self.value >= Stage.DEWARPED.value


class Artifact(enum.Enum):
	SEGMENTATION = ("segment.zip",)
	DEWARPING_TRANSFORM = ("dewarp.zip",)
	TABLES = ("tables.json",)
	ORDER = ("order.json",)
	OCR = ("ocr.zip",)
	COMPOSE = ("compose.zip",)
	RUNTIME = ("runtime.json",)
	SIGNATURE = ("signature.zip",)
	THUMBNAIL = ("thumbnail.jpg",)
	CONTOURS = ("contours.%s.zip", {
		Stage.WARPED: 0,
		Stage.DEWARPED: 1,
		Stage.AGGREGATE: 2,
		Stage.RELIABLE: 3})
	LINES = ("lines.%s.zip", {
		Stage.WARPED: 0,
		#Stage.DEWARPED  # not supported
		#Stage.AGGREGATE  # not supported
		Stage.RELIABLE: 3  # not supported
	})

	# for debugging.
	DINGLEHOPPER = ("dinglehopper.xml")

	def __init__(self, filename, stages=None):
		self._filename = filename
		self._stages = stages

	@property
	def stages(self):
		if self._stages:
			return self._stages.keys()
		else:
			return None

	def filename(self, stage=None):
		s = self._filename
		if self._stages is not None:
			if stage is None:
				raise RuntimeError(
					"need to specify stage for loading %s" % self)
			variant = self._stages.get(stage)
			if variant is None:
				raise RuntimeError(
					"%s is not supported for stage %s" % (self, stage))
			s = s % str(variant)
		return s


class DebuggingArtifact:
	def __init__(self, filename):
		self._filename = filename

	def filename(self, stage=None):
		return self._filename


class Annotation(DebuggingArtifact):
	def __init__(self, name):
		filename = "annotation.%s.jpg" % name
		super().__init__(filename)


Contours = collections.namedtuple("Contours", ["items", "meta"])


def read_contours(path: Path, pred_type, open=open):
	items = []
	pred_meta = dict()

	with open(path, "rb") as f:
		with zipfile.ZipFile(f, "r") as zf:
			meta = json.loads(zf.read("meta.json"))

			def filter_path(contours_path):
				prediction_name = contours_path[0]
				t = PredictorType[meta[prediction_name]["type"]]
				return t == pred_type

			for name in zf.namelist():
				if name.endswith("/meta.json"):
					parts = tuple(name.split("/"))
					if filter_path(parts):
						pred_meta[tuple(parts[:-1])] = json.loads(zf.read(name))

				if not name.endswith(".wkt"):
					continue

				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				if not filter_path(parts):
					continue

				items.append((
					parts,
					shapely.wkt.loads(zf.read(name).decode("utf8"))))

	return Contours(items, pred_meta)


def read_separators(path: Path, open=open):
	contours = read_contours(
		path, PredictorType.SEPARATOR, open=open)

	separators = dict()
	for sep_path, polygon in contours.items:
		separators[sep_path] = polygon

	widths = dict()
	if contours.meta:
		for k, data in contours.meta.items():
			for i, w in enumerate(data["width"]):
				widths[k + (str(i),)] = w

	return separators, widths


class Regions:
	def __init__(self, path: Path, page, stage, open=open):
		blocks = dict()

		for parts, polygon in read_contours(
			path,
			PredictorType.REGION,
			open=open).items:

			blocks[parts] = Block(
				page, polygon, stage)

		self._blocks = blocks

	@property
	def by_path(self):
		return self._blocks

	@cached_property
	def by_predictors(self):
		by_predictors = collections.defaultdict(list)
		for k, block in self._blocks.items():
			by_predictors[k[:2]].append(block)
		return by_predictors


class Lines:
	def __init__(self, path: Path, regions, stage=Stage.WARPED, open=open):
		blocks = regions.by_path
		assert all(block.stage == stage for block in blocks.values())
		self._meta = None
		lines = dict()
		with open(path, "rb") as lf:
			with zipfile.ZipFile(lf, "r") as zf:
				for name in zf.namelist():
					if name == "meta.json":
						self._meta = json.loads(zf.read(name))
						continue
					if not name.endswith(".json"):
						raise RuntimeError("illegal file %s in %s." % (
							name, path))
					stem = name.rsplit('.', 1)[0]
					parts = tuple(stem.split("/"))
					block = blocks[tuple(parts[:3])]
					line_info = json.loads(zf.read(name))
					lines[parts] = Line(block, **line_info)
		self._lines = lines

	@property
	def meta(self):
		return self._meta

	@property
	def min_confidence(self):
		return self.meta.get("min_confidence", 0.5)

	@property
	def by_path(self):
		return self._lines


class Reader:
	def __init__(self, artifacts, stage, page_path, take_any, open=open):
		artifacts = set(artifacts)

		if Artifact.LINES in artifacts:
			artifacts.add(Artifact.CONTOURS)
		if Artifact.CONTOURS in artifacts:
			artifacts.add(Artifact.SEGMENTATION)
		if stage and stage.is_dewarped and Artifact.CONTOURS in artifacts:
			artifacts.add(Artifact.DEWARPING_TRANSFORM)

		self._artifacts = artifacts
		self._stage = stage
		self._page_path = page_path
		self._data_path = find_data_path(page_path)
		self._take_any = take_any
		self._open = open

	@property
	def data_path(self):
		return self._data_path

	@property
	def paths(self):
		return [self.path(a) for a in self._artifacts]

	def path(self, artifact):
		if artifact not in self._artifacts:
			raise RuntimeError("read on undeclared %s" % artifact)
		return self._data_path / artifact.filename(self._stage)

	def is_ready(self):
		if self._take_any:
			return True
		else:
			return all(p.exists() for p in self.paths)

	@property
	def missing(self):
		return [p for p in self.paths if not p.exists()]

	def load_json(self, artifact):
		with open(self.path(artifact), "r") as f:
			return json.loads(f.read())

	@cached_property
	def page(self):
		if self._stage.is_dewarped:
			return Page(self._page_path, self.dewarping_transform)
		else:
			return Page(self._page_path)

	@cached_property
	def predictors(self):
		from origami.core.segment import Segmentation
		return Segmentation.read_predictors(self.path(Artifact.SEGMENTATION))

	@cached_property
	def _segmentation(self):
		from origami.core.segment import Segmentation
		return Segmentation.open(self.path(Artifact.SEGMENTATION))

	@cached_property
	def segmentation(self):
		assert self._stage is None or self._stage == Stage.WARPED
		return self._segmentation

	@cached_property
	def regions(self):
		return Regions(
			self.path(Artifact.CONTOURS),
			self.page,
			self._stage,
			open=self._open)

	@cached_property
	def separators(self):
		geoms, widths = read_separators(
			self.path(Artifact.CONTOURS),
			open=self._open)
		return Separators(
			self._segmentation,
			geoms,
			widths)

	@cached_property
	def lines(self):
		return Lines(
			self.path(Artifact.LINES),
			self.regions,
			self._stage,
			open=self._open)

	@cached_property
	def dewarping_transform(self):
		from origami.core.dewarp import Grid
		return Grid.open(self.path(Artifact.DEWARPING_TRANSFORM))

	@cached_property
	def tables(self):
		return self.load_json(Artifact.TABLES)

	@cached_property
	def order(self):
		return self.load_json(Artifact.ORDER)

	@property
	def signature(self):
		return zipfile.ZipFile(self.path(Artifact.SIGNATURE), "r")

	@property
	def compose(self):
		return zipfile.ZipFile(self.path(Artifact.COMPOSE), "r")

	@cached_property
	def ocr(self):
		texts = dict()
		with zipfile.ZipFile(self.path(Artifact.OCR), "r") as zf:
			for k in zf.namelist():
				texts[k] = zf.read(k).decode("utf8")
		return texts

	@cached_property
	def sorted_ocr(self):
		def sortable_path(line_name):
			line_path = tuple(line_name.rsplit(".", 1)[0].split("/"))
			return line_path[:-1] + (int(line_path[-1]),)

		keys = sorted(map(sortable_path, self.ocr.keys()))
		for path in keys:
			filename = "/".join(map(str, path)) + ".txt"
			yield tuple(map(str, path)), self.ocr[filename]

	@property
	def annotation(self):
		assert len(self._artifacts) == 1
		annotation = list(self._artifacts)[0]
		assert isinstance(annotation, Annotation)
		return self.path(annotation)


class Input:
	def __init__(self, *artifacts, stage=None, take_any=False):
		assert all(isinstance(x, (Artifact, DebuggingArtifact)) for x in artifacts)
		self._artifacts = set(artifacts)
		self._stage = stage
		self._take_any = take_any

	def instantiate(self, processor, overwrite, **kwargs):
		return Reader(
			self._artifacts, self._stage,
			take_any=self._take_any,
			open=processor.lock_or_open, **kwargs)


class Writer:
	def __init__(self, artifacts, stage, page_path, processor, overwrite):
		self._artifacts = artifacts
		self._stage = stage
		self._page_path = page_path
		self._data_path = find_data_path(page_path)
		self._processor = processor
		self._overwrite = overwrite

	@property
	def paths(self):
		return [self.path(a) for a in self._artifacts]

	def path(self, artifact):
		if artifact not in self._artifacts:
			raise RuntimeError("write on undeclared %s" % artifact)
		return self._data_path / artifact.filename(self._stage)

	def is_ready(self):
		return self._overwrite or not any(p.exists() for p in self.paths)

	@property
	def missing(self):
		return []

	def write_json(self, artifact, data):
		path = self.path(artifact)
		with atomic_write(path, mode="wb", overwrite=self._overwrite) as f:
			f.write(json.dumps(data).encode("utf8"))

	def write_zip_file(self, artifact):
		return self._processor.write_zip_file(
			self.path(artifact),
			overwrite=self._overwrite)

	def segmentation(self, segmentation):
		path = self.path(Artifact.SEGMENTATION)
		with atomic_write(path, mode="wb", overwrite=self._overwrite) as f:
			segmentation.save(f)

	@contextmanager
	def contours(self, copy_meta_from=None):
		with self.write_zip_file(Artifact.CONTOURS) as f:

			if copy_meta_from is not None:
				path = copy_meta_from.path(Artifact.CONTOURS)
				with zipfile.ZipFile(path, "r") as zf:
					f.writestr("meta.json", zf.read("meta.json"))
					for name in zf.namelist():
						if name.endswith("/meta.json"):
							f.writestr(name, zf.read(name))

			yield f

	def lines(self):
		return self.write_zip_file(Artifact.LINES)

	def ocr(self):
		return self.write_zip_file(Artifact.OCR)

	@contextmanager
	def dewarping_transform(self):
		path = self.path(Artifact.DEWARPING_TRANSFORM)
		with atomic_write(path, mode="wb", overwrite=self._overwrite) as f:
			yield f

	def tables(self, data):
		self.write_json(Artifact.TABLES, data)

	def order(self, data):
		self.write_json(Artifact.ORDER, data)

	def compose(self):
		return self.write_zip_file(Artifact.COMPOSE)

	def signature(self):
		return self.write_zip_file(Artifact.SIGNATURE)

	def annotation(self, image):
		assert len(self._artifacts) == 1
		annotation = list(self._artifacts)[0]
		assert isinstance(annotation, Annotation)
		image.save(str(self.path(annotation)))


class Output:
	def __init__(self, *artifacts, stage=None):
		assert all(isinstance(x, (Artifact, DebuggingArtifact)) for x in artifacts)
		self._artifacts = set(artifacts)
		self._stage = stage

	def instantiate(self, **kwargs):
		return Writer(self._artifacts, self._stage, **kwargs)


def parse_artifact(name):
	if "/" in name:
		parts = list(map(
			lambda s: s.strip().upper(), name.split("/")))
		if len(parts) != 2:
			raise ValueError(name)
		t1, t2 = parts
		if t1 in Artifact:
			artifact = Artifact[t1]
			stage = Stage[t2]
			return artifact, stage
		elif t1 == "ANNOTATION":
			artifact = Annotation(t2.lower())
		else:
			raise ValueError(name)
	else:
		try:
			artifact = Artifact[name.upper()]
		except KeyError:
			raise click.UsageError(
				"illegal artifact name %s" % name)

	return artifact, None

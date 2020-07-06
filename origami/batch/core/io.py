#!/usr/bin/env python3

import enum
import json
import zipfile
import shapely.wkt

from pathlib import Path
from contextlib import contextmanager
from cached_property import cached_property
from atomicwrites import atomic_write

from origami.core.page import Page
from origami.core.predict import PredictorType
from origami.core.block import Block, Line


def find_data_path(page_path):
	return page_path.with_suffix(".out")


class Stage(enum.Enum):
	WARPED = 0
	DEWARPED = 1
	AGGREGATE = 2
	RELIABLE = 3

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
	CONTOURS = ("contours.%s.zip", {
		Stage.WARPED: 0,
		Stage.DEWARPED: 1,
		Stage.AGGREGATE: 2,
		Stage.RELIABLE: 3})
	LINES = ("lines.%s.zip", {
		Stage.WARPED: 0,
		# Stage.DEWARPED: None,
		Stage.AGGREGATE: 2,
		Stage.RELIABLE: 2})

	def __init__(self, filename, stages=None):
		self._filename = filename
		self._stages = stages

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


def read_contours(path: Path, pred_type, open=open):
	with open(path, "rb") as f:
		with zipfile.ZipFile(f, "r") as zf:
			meta = json.loads(zf.read("meta.json"))

			for name in zf.namelist():
				if not name.endswith(".wkt"):
					continue

				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				prediction_name = parts[0]

				t = PredictorType[meta[prediction_name]["type"]]
				if t != pred_type:
					continue

				yield parts, shapely.wkt.loads(zf.read(name).decode("utf8"))


def read_separators(path: Path, open=open):
	separators = dict()

	for parts, polygon in read_contours(
		path, PredictorType.SEPARATOR, open=open):
		separators[parts] = polygon

	return separators


def read_lines(path: Path, blocks, stage=Stage.WARPED, open=open):
	assert all(block.stage == stage for block in blocks.values())
	lines = dict()
	with open(path, "rb") as lf:
		with zipfile.ZipFile(lf, "r") as zf:
			for name in zf.namelist():
				if name == "meta.json":
					continue
				if not name.endswith(".json"):
					raise RuntimeError("illegal file %s in %s." % (
						name, page_path.with_suffix(".lines.zip")))
				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				block = blocks[tuple(parts[:3])]
				line_info = json.loads(zf.read(name))
				lines[parts] = Line(block, **line_info)
	return lines


class Reader:
	def __init__(self, artifacts, stage, page_path, open=open):
		artifacts = set(artifacts)

		if Artifact.LINES in artifacts:
			artifacts.add(Artifact.CONTOURS)
		if stage and stage.is_dewarped and Artifact.CONTOURS in artifacts:
			artifacts.add(Artifact.DEWARPING_TRANSFORM)

		self._artifacts = artifacts
		self._stage = stage
		self._page_path = page_path
		self._data_path = find_data_path(page_path)
		self._open = open

	@property
	def paths(self):
		return [self.path(a) for a in self._artifacts]

	def path(self, artifact):
		if artifact not in self._artifacts:
			raise RuntimeError("read on undeclared %s" % artifact)
		return self._data_path / artifact.filename(self._stage)

	def is_ready(self):
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
	def segmentation(self):
		assert self._stage is None or self._stage == Stage.WARPED
		from origami.core.segment import Segmentation
		return Segmentation.open(self.path(Artifact.SEGMENTATION))

	@cached_property
	def blocks(self):
		blocks = dict()
		page = self.page

		for parts, polygon in read_contours(
			self.path(Artifact.CONTOURS),
			PredictorType.REGION,
			open=self._open):

			blocks[parts] = Block(
				page, polygon, self._stage)

		return blocks

	@cached_property
	def separators(self):
		return read_separators(
			self.path(Artifact.CONTOURS),
			open=self._open)

	@cached_property
	def lines(self):
		return read_lines(
			self.path(Artifact.LINES),
			self.blocks,
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

	@cached_property
	def ocr(self):
		texts = dict()
		with zipfile.ZipFile(self.path(Artifact.OCR), "r") as zf:
			for k in zf.namelist():
				texts[k] = zf.read(k).decode("utf8")
		return texts


class Input:
	def __init__(self, *artifacts, stage=None):
		assert all(isinstance(x, (Artifact, DebuggingArtifact)) for x in artifacts)
		self._artifacts = set(artifacts)
		self._stage = stage

	def instantiate(self, processor, overwrite, **kwargs):
		return Reader(self._artifacts, self._stage, open=processor.lock, **kwargs)


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

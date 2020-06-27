import imghdr
import click
import zipfile
import io
import json
import numpy as np
import PIL.Image
import scipy.ndimage
import skimage.morphology

from pathlib import Path
from atomicwrites import atomic_write
from ast import literal_eval as make_tuple
from functools import partial

from origami.batch.core.processor import Processor

from origami.core.segment import Segmentation
from origami.core.page import Page, Annotations
import origami.core.contours as contours
from origami.core.block import Block, Stage
from origami.core.predict import PredictorType


class ContoursProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def _process_region_contours(self, zf, annotations, prediction):
		pipeline = [
			contours.Contours(),
			contours.Decompose(),
			contours.FilterByArea(annotations.magnitude * self._options["region_minsize"])
		]

		region_contours = annotations.create_multi_class_contours(
			prediction.labels,
			contours.fold_operator([
				contours.multi_class_constructor(
					pipeline=pipeline,
					classes=[c for c in prediction.classes if c != prediction.classes.BACKGROUND]),
				contours.HeuristicFrameDetector(
					annotations.size, self._options["margin_noise"]).multi_class_filter
			]))

		for prediction_class, shapes in region_contours.items():
			for region_id, polygon in enumerate(shapes):
				block = Block(annotations.page, polygon, stage=Stage.WARPED)

				if self._options["export_images"]:
					with io.BytesIO() as f:
						im, _ = block.extract_image()
						im.save(f, format='png')
						data = f.getvalue()

					zf.writestr("%s/%s/%03d.png" % (
						prediction.name, prediction_class.name, region_id), data)

				zf.writestr("%s/%s/%03d.wkt" % (
					prediction.name, prediction_class.name, region_id), polygon.wkt)

	def _process_separator_contours(self, zf, annotations, prediction):

		def build_pipeline(label_class):
			return [
				contours.Contours(),
				contours.Simplify(0),
				contours.EstimatePolyline(label_class.orientation.direction),
				contours.Simplify(annotations.magnitude * self._options["sep_threshold"])
			]

		region_separators = annotations.create_multi_class_contours(
			prediction.labels,
			contours.multi_class_constructor(
				pipeline=build_pipeline,
				classes=[c for c in prediction.classes if c != prediction.classes.BACKGROUND]))

		for prediction_class, shapes in region_separators.items():
			widths = []
			for separator_id, polyline in enumerate(shapes):
				zf.writestr("%s/%s/%03d.wkt" % (
					prediction.name, prediction_class.name, separator_id), polyline.line_string.wkt)
				widths.append(polyline.width)

			zf.writestr("%s/%s/meta.json" % (
				prediction.name, prediction_class.name), json.dumps(dict(width=widths)))

	def should_process(self, p: Path) -> bool:
		return (imghdr.what(p) is not None) and\
			p.with_suffix(".segment.zip").exists() and\
			not p.with_suffix(".warped.contours.zip").exists()

	def process(self, p: Path):
		segmentation = Segmentation.open(p.with_suffix(".segment.zip"))

		page = Page(p)
		annotations = Annotations(page, segmentation)

		handlers = dict((
			(PredictorType.REGION, self._process_region_contours),
			(PredictorType.SEPARATOR, self._process_separator_contours)
		))

		zf_path = p.with_suffix(".warped.contours.zip")
		with atomic_write(zf_path, mode="wb", overwrite=False) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				info = dict(version=1)
				for prediction in segmentation.predictions:
					handlers[prediction.type](zf, annotations, prediction)
					info[prediction.name] = dict(type=prediction.type.name)
				zf.writestr("meta.json", json.dumps(info))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-x', '--export-images',
	is_flag=True,
	default=False,
	help="Export region images (larger files).")
@click.option(
	'-r', '--region-minsize',
	type=float,
	default=0.01,  # might be a single word.
	help="Ignore regions below this relative size.")
@click.option(
	'-m', '--margin-noise',
	type=float,
	default=0.05,
	help="Max. relative width of margin noise.")
@click.option(
	'-s', '--sep-threshold',
	type=float,
	default=4 / 1000,
	help="Simplification of separator polylines.")
@click.option(
	'--ink-spread',
	type=str,
	default="(20, 20)",
	help="Ink dilation for whitespace detection.")
@click.option(
	'--ink-opening',
	type=str,
	default="(5, 5)",
	help="Opening amount to remove ink overflow between columns.")
@click.option(
	'--region-glue',
	type=float,
	default=0.02,
	help="Threshold for considering region as glue blob.")
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
def extract_contours(data_path, **kwargs):
	""" Extract contours from all document images in DATA_PATH.
	Information from segmentation and binarize batch needs to be present. """
	processor = ContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	extract_contours()

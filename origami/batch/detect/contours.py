#!/usr/bin/env python3

import click
import io
import json

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output

from origami.core.page import Page, Annotations
import origami.core.contours as contours
from origami.core.block import Block
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
			contours.FilterByArea(annotations.geometry.rel_area(self._options["region_area"]))
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

					zf.writestr("%s/%s/%d.png" % (
						prediction.name, prediction_class.name, region_id), data)

				zf.writestr("%s/%s/%d.wkt" % (
					prediction.name, prediction_class.name, region_id), polygon.wkt)

	def _process_separator_contours(self, zf, annotations, prediction):

		def build_pipeline(label_class):
			return [
				contours.Contours(),
				contours.Simplify(0),
				contours.EstimatePolyline(label_class.orientation.direction),
				contours.Simplify(annotations.geometry.rel_length(
					self._options["separator_threshold"]))
			]

		region_separators = annotations.create_multi_class_contours(
			prediction.labels,
			contours.multi_class_constructor(
				pipeline=build_pipeline,
				classes=[c for c in prediction.classes if c != prediction.classes.BACKGROUND]))

		for prediction_class, shapes in region_separators.items():
			widths = []
			for separator_id, polyline in enumerate(shapes):
				zf.writestr("%s/%s/%d.wkt" % (
					prediction.name, prediction_class.name, separator_id), polyline.line_string.wkt)
				widths.append(polyline.width)

			zf.writestr("%s/%s/meta.json" % (
				prediction.name, prediction_class.name), json.dumps(dict(width=widths)))

	def artifacts(self):
		return [
			("input", Input(Artifact.SEGMENTATION)),
			("output", Output(Artifact.CONTOURS, stage=Stage.WARPED))
		]

	def process(self, p: Path, input, output):
		segmentation = input.segmentation

		page = Page(p)
		annotations = Annotations(page, segmentation)

		handlers = dict((
			(PredictorType.REGION, self._process_region_contours),
			(PredictorType.SEPARATOR, self._process_separator_contours)
		))

		with output.contours() as zf:
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
	'--export-images',
	is_flag=True,
	default=False,
	help="Export region images (larger files).")
@click.option(
	'--region-area',
	type=float,
	default=0.01,  # might be a single word.
	help="Ignore regions below this relative size.")
@click.option(
	'--margin-noise',
	type=float,
	default=0.075,
	help="Max. relative width of margin noise.")
@click.option(
	'--separator-threshold',
	type=float,
	default=4 / 1000,
	help="Simplification of separator polylines.")
@Processor.options
def extract_contours(data_path, **kwargs):
	""" Extract contours from all document images in DATA_PATH.
	Information from segmentation and binarize batch needs to be present. """
	processor = ContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	extract_contours()

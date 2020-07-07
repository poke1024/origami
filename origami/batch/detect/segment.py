#!/usr/bin/env python3

import click

from pathlib import Path

from origami.core.segment import SegmentationPredictor
from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


class SegmentationProcessor(Processor):
	def __init__(self, predictor, options):
		super().__init__(options)
		self._predictor = predictor

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("output", Output(Artifact.SEGMENTATION)),
		]

	def process(self, p: Path, output):
		segmentation = self._predictor(p)
		output.segmentation(segmentation)


@click.command()
@click.option(
	'-m', '--model',
	required=True,
	type=click.Path(exists=True),
	help='path to prediction model')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def segment(data_path, model, **kwargs):
	""" Perform page segmentation on all document images in DATA_PATH. """
	predictor = SegmentationPredictor(model)
	processor = SegmentationProcessor(predictor, kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()


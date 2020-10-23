#!/usr/bin/env python3

import click

from pathlib import Path

from origami.core.segment import SegmentationPredictor
from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


class SegmentationProcessor(Processor):
	def __init__(self, model, options):
		super().__init__(options)
		self._model_path = model
		self._options = options
		self._predictor = None

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("output", Output(Artifact.SEGMENTATION)),
		]

	def process(self, p: Path, output):
		if self._predictor is None:
			self._predictor = SegmentationPredictor(
				self._model_path,
				grayscale=self._options["grayscale"],
				target=self._options["target"])

		segmentation = self._predictor(p)
		output.segmentation(segmentation)


@click.command()
@click.option(
	'-m', '--model',
	required=True,
	type=click.Path(exists=True),
	help='path to prediction model')
@click.option(
	'-t', '--target',
	required=False,
	type=str,
	default='quality',
	help='configure speed vs quality')
@click.option(
	'--grayscale',
	is_flag=True,
	default=False,
	help='treat input images as grayscale')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def segment(data_path, model, **kwargs):
	""" Perform page segmentation on all document images in DATA_PATH. """
	processor = SegmentationProcessor(model, kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()


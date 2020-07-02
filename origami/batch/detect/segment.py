import imghdr
import click
import re

from pathlib import Path
from atomicwrites import atomic_write

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
			("output", Output(Artifact.SEGMENT)),
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
def segment(data_path, model, **kwargs):
	""" Perform page segmentation on all document images in DATA_PATH. """
	predictor = SegmentationPredictor(model)
	processor = SegmentationProcessor(predictor, kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()


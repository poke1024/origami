#!/usr/bin/env python3

import imghdr
import click
import PIL.Image
import logging

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation
from origami.batch.annotate.utils import render_blocks, render_separators, qt_app


class AnnotateContoursProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._stage = Stage[options["stage"].upper()]

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.SEGMENTATION, stage=Stage.WARPED)),
			("input", Input(Artifact.CONTOURS, stage=self._stage)),
			("output", Output(Annotation("contours." + self._stage.name.lower()))),
		]

	def process(self, page_path: Path, warped, input, output):
		blocks = input.blocks
		separators = input.separators

		if not blocks:
			logging.info("no blocks for %s" % page_path)
			return

		page = list(blocks.values())[0].page
		predictors = warped.predictors

		qt_im = ImageQt(page.dewarped if self._stage.is_dewarped else page.warped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		def get_label(block_path):
			classifier, segmentation_label, block_id = block_path
			return (classifier, segmentation_label), int(block_id.split(".")[0])

		pixmap = render_blocks(pixmap, blocks, get_label, predictors)
		pixmap = render_separators(pixmap, separators)

		output.annotation(pixmap.toImage())


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--stage',
	type=str,
	default="warped")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
@click.option(
	'--overwrite',
	is_flag=True,
	default=False,
	help="Recompute and overwrite existing result files.")
def debug_contours(data_path, **kwargs):
	""" Export annotate information on vectors for all document images in DATA_PATH. """
	processor = AnnotateContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = qt_app()
	debug_contours()

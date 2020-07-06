#!/usr/bin/env python3

import imghdr
import click
import PIL.Image

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.processor import Processor
from origami.batch.annotate.utils import render_lines, qt_app
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation


class DebugWarpProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.CONTOURS, Artifact.LINES, stage=Stage.WARPED)),
			("output", Output(Annotation("warp"))),
		]

	def process(self, page_path: Path, warped, output):
		lines = warped.lines

		def get_label(lines_path):
			classifier, segmentation_label, block_id, line_id = lines_path
			return (classifier, segmentation_label), line_id

		qt_im = ImageQt(PIL.Image.open(page_path))
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_lines(pixmap, lines, get_label)
		output.annotation(pixmap.toImage())


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
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
def debug_warp(data_path, **kwargs):
	""" Export annotate information on lines for all document images in DATA_PATH. """
	processor = DebugWarpProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = qt_app()
	debug_warp()

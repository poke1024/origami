#!/usr/bin/env python3

import imghdr
import click
import PIL.Image
import importlib

from pathlib import Path


if importlib.util.find_spec("PySide2"):
	from PySide2 import QtGui
else:
	from PySide6 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.processor import Processor
from origami.batch.annotate.utils import render_lines, render_paths
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation


class DebugFlowProcessor(Processor):
	def __init__(self, options):
		super().__init__(options, needs_qt=True)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(
				Artifact.SEGMENTATION,
				Artifact.CONTOURS, Artifact.LINES, stage=Stage.WARPED)),
			("output", Output(Annotation("flow"))),
		]

	def process(self, page_path: Path, warped, output):
		lines = warped.lines.by_path

		qt_im = ImageQt(PIL.Image.open(page_path))
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_lines(
			pixmap, lines, predictors=warped.predictors, show_vectors=True)

		for k, v in warped.separators.by_path.items():
			colors = dict(T="yellow", H="blue", V="red")
			path = list(v.coords)
			pixmap = render_paths(
				pixmap, [path], colors[k[1]], opacity=0.9, show_dir=True)

		output.annotation(pixmap.toImage())


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def debug_flow(data_path, **kwargs):
	""" Annotate information on flow for all document images in DATA_PATH. """
	processor = DebugFlowProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	debug_flow()

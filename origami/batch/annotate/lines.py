#!/usr/bin/env python3

import click
import PIL.Image
import io
import numpy as np
import skimage.filters
import PIL.ImageEnhance
import importlib

from pathlib import Path


if importlib.util.find_spec("PySide2"):
	from PySide2 import QtGui
else:
	from PySide6 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.processor import Processor
from origami.batch.annotate.utils import render_warped_line_paths, render_warped_line_confidence
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation


class DebugLinesProcessor(Processor):
	def __init__(self, options):
		super().__init__(options, needs_qt=True)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("warped", Input(Artifact.SEGMENTATION)),
			("reliable", Input(
				Artifact.CONTOURS, Artifact.LINES, stage=Stage.RELIABLE)),
			("output", Output(Annotation("lines")))
		]

	def process(self, page_path: Path, warped, reliable, output):
		blocks = reliable.regions.by_path
		lines = reliable.lines.by_path

		if not blocks:
			return

		page = reliable.page
		predictors = warped.predictors

		qt_im = ImageQt(page.warped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)
		pixmap = render_warped_line_paths(
			pixmap, lines, predictors, opacity=1)

		buffer = QtCore.QBuffer()
		buffer.open(QtCore.QBuffer.ReadWrite)
		pixmap.toImage().save(buffer, "PNG")
		im = PIL.Image.open(io.BytesIO(buffer.data())).convert("RGB")

		binarized = np.array(page.warped)
		thresh = skimage.filters.threshold_sauvola(
			binarized, window_size=15)
		binarized = PIL.Image.fromarray(
			(binarized > thresh).astype(np.uint8) * 255)

		alpha = np.array(
			binarized.convert("L")).astype(np.float32)
		alpha += self._options["marker_opacity"]
		alpha = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)

		im = PIL.Image.composite(
			binarized.convert("RGB"), im, PIL.Image.fromarray(1 - alpha))

		qt_im = ImageQt(im)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		if self._options["show_confidence"]:
			pixmap = render_warped_line_confidence(pixmap, lines)

		output.annotation(pixmap.toImage())



@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--show-confidence',
	is_flag=True,
	default=False,
	help='annotate confidence for suspicious lines')
@click.option(
	'--marker-opacity',
	type=float,
	default=0.5,
	help='opacity of line markers')
@Processor.options
def debug_lines(data_path, **kwargs):
	""" Export annotate information on lines for all document images in DATA_PATH. """
	processor = DebugLinesProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	debug_lines()

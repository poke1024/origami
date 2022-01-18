#!/usr/bin/env python3

import click
import PIL.Image
import importlib

from pathlib import Path
from PIL.ImageQt import ImageQt


if importlib.util.find_spec("PySide2"):
	from PySide2 import QtGui
else:
	from PySide6 import QtGui

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, Annotation


class DebugGridProcessor(Processor):
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
				Artifact.CONTOURS, Artifact.LINES,
				Artifact.DEWARPING_TRANSFORM, stage=Stage.WARPED)),
			("output", Output(Annotation("grid"))),
		]

	def process(self, page_path: Path, warped, output):
		grid = warped.dewarping_transform
		pts = grid.points()

		qt_im = ImageQt(PIL.Image.open(page_path))
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		grid_n = 50

		qp = QtGui.QPainter()
		qp.begin(pixmap)

		try:
			qp.setBrush(QtGui.QBrush(QtGui.QColor("white")))
			qp.setOpacity(0.5)
			qp.drawRect(0, 0, pixmap.width(), pixmap.height())

			pen = QtGui.QPen()
			pen.setWidth(10)
			pen.setColor(QtGui.QColor.fromHsv(200, 255, 128))
			pen.setCapStyle(QtCore.Qt.SquareCap)

			qp.setPen(pen)
			qp.setOpacity(0.75)

			for y in range(0, pts.shape[0], int(pixmap.height() / grid_n)):
				qp.drawPolyline([QtCore.QPointF(*p) for p in pts[y, :]])

			for x in range(0, pts.shape[1], int(pixmap.width() / grid_n)):
				qp.drawPolyline([QtCore.QPointF(*p) for p in pts[:, x]])

		finally:
			qp.end()

		output.annotation(pixmap.toImage())


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def debug_grid(data_path, **kwargs):
	""" Annotate information on dewarping grid for all document images in DATA_PATH. """
	processor = DebugGridProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	debug_grid()

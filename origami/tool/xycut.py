import sys
import click
import cv2
import PIL.Image
from PIL.ImageQt import ImageQt

from PySide2 import QtWidgets, QtGui, QtCore
from pathlib import Path
from functools import lru_cache

from origami.batch.core.utils import *
from origami.batch.core.deskew import Deskewer
from origami.batch.annotate.utils import render_blocks, default_pen

import origami.core.xycut as xycut


class Canvas(QtWidgets.QScrollArea):
	def __init__(self, page_path):
		super().__init__()

		self.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum,
			QtWidgets.QSizePolicy.Minimum)
		self.setMinimumSize(1280, 600)

		self.label = QtWidgets.QLabel()

		self.setWidgetResizable(True)
		self.setWidget(self.label)

		blocks = read_blocks(page_path)
		lines = read_lines(page_path, blocks)

		deskewer = Deskewer(lines)

		im = PIL.Image.open(page_path)
		qt_im = ImageQt(deskewer.image(im))
		self._pixmap = QtGui.QPixmap.fromImage(qt_im)

		boxes = []
		for block_path, block in blocks.items():
			minx, miny, maxx, maxy = deskewer.shapely(block.image_space_polygon).bounds
			boxes.append(xycut.Box(block_path, minx, miny, maxx, maxy))
		self._boxes = boxes

		self._zoom = 2
		self._split_path = None
		self.setSplitPath([])

	def setSplitPath(self, path):
		self._split_path = None if path is None else tuple(path)
		self.updatePixmap()

	def cutsForPath(self, path):
		cuts = []
		boxes = self._boxes
		for k in path + (None,):
			cut = xycut.XYCut(boxes)
			cuts.append(cut)
			if k is None or not cut.valid:
				break
			boxes = cut[k]
		return cuts

	@lru_cache(maxsize=16)
	def annotatedPixmap(self, path, zoom):
		pixmap = self._pixmap.copy()

		if path is not None:
			cuts = self.cutsForPath(path)

			qp = QtGui.QPainter()
			qp.begin(pixmap)

			try:
				qp.setOpacity(0.5)
				qp.setPen(default_pen("blue"))

				for box in self._boxes:
					minx, miny, maxx, maxy = box.bounds
					qp.drawRect(minx, miny, maxx - minx, maxy - miny)

				for i, cut in enumerate(cuts):
					if i < len(cuts) - 1:
						qp.setOpacity(0.25)
						qp.setPen(default_pen("red", width=10))
					else:
						qp.setOpacity(0.75)
						qp.setPen(default_pen("red", width=20))

					if cut.valid:
						if cut.axis == 0:
							coords = [(cut.x, 0), (cut.x, self._pixmap.height())]
							#print("(%d) x split at %.2f" % (i + 1, cut.x))
						else:
							coords = [(0, cut.x), (self._pixmap.width(), cut.x)]
							#print("(%d) y split at %.2f" % (i + 1, cut.x))

						qp.drawPolyline([QtCore.QPointF(*p) for p in coords])

				if cuts[-1].valid:
					qp.setOpacity(0.5)
					qp.setPen(default_pen("blue"))

					for i, boxes in enumerate(cuts[-1]):
						qp.setBrush(QtGui.QBrush(
							QtGui.QColor.fromHsv(100 * i, 200, 150)))

						for box in boxes:
							minx, miny, maxx, maxy = box.bounds
							qp.drawRect(minx, miny, maxx - minx, maxy - miny)
			finally:
				qp.end()

		z = 2 ** -zoom
		w = pixmap.width()
		h = pixmap.height()

		return pixmap.scaled(
			int(w * z), int(h * z),
			aspectMode=QtCore.Qt.KeepAspectRatio,
			transformMode=QtCore.Qt.SmoothTransformation)

	def updatePixmap(self):
		pixmap = self.annotatedPixmap(
			self._split_path, self._zoom)

		self.label.setPixmap(pixmap)

		self.label.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum,
			QtWidgets.QSizePolicy.Minimum)
		self.label.setMinimumSize(
			pixmap.width(), pixmap.height())

	def keyPressEvent(self, event):
		s = event.text()
		if s == "+":
			self._zoom = max(self._zoom - 1, 0)
		elif s == "-":
			self._zoom = min(self._zoom + 1, 10)
		self.updatePixmap()


class Form(QtWidgets.QDialog):
	def __init__(self, page_path, parent=None):
		super().__init__(parent)
		self.page_path = page_path
		self.setWindowTitle(page_path.name)

		self.canvas = Canvas(page_path)
		self.edit = QtWidgets.QLineEdit("")

		self.edit.textChanged.connect(self.editChanged)

		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.canvas)
		layout.addWidget(self.edit)

		self.setLayout(layout)

	def editChanged(self):
		text = self.edit.text()
		sel = {"0": 0, "1": 1}
		path = [sel.get(c, -1) for c in text]
		if -1 in path:
			path = None  # invalid input.
		self.canvas.setSplitPath(path)


@click.command()
@click.argument(
	'page_path',
	type=click.Path(exists=True),
	required=True)
def app(page_path, **kwargs):
	""" Debug xycut for page at PAGE_PATH. """
	app = QtWidgets.QApplication(sys.argv)

	form = Form(Path(page_path))
	form.show()

	sys.exit(app.exec_())


if __name__ == '__main__':
	app()

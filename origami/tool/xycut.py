#!/usr/bin/env python3

import sys
import click
import cv2
import imghdr
import PIL.Image
from PIL.ImageQt import ImageQt

from PySide2 import QtWidgets, QtGui, QtCore
from pathlib import Path
from functools import lru_cache

from origami.batch.core.deskew import Deskewer
from origami.batch.core.io import Reader, Artifact, Stage
from origami.batch.annotate.utils import render_blocks, default_pen

from origami.core.page import Page
from origami.core.segment import Segmentation
from origami.core.separate import Separators, ObstacleSampler
import origami.core.xycut as xycut


class Configuration:
	def __init__(self, page_path):
		self._page_path = page_path
		self._image = None
		self._polygons = dict()

	@property
	def image(self):
		return self._image

	@property
	def polygons(self):
		return self._polygons

	@property
	def xycut_score(self):
		return "longest_cut"


class DeskewedConfiguration(Configuration):
	def __init__(self, page_path):
		super().__init__(page_path)

		warped = Reader([Artifact.CONTOURS.Artifact.LINES], Stage.WARPED, self._page_path)
		blocks = warped.regions.by_path
		lines = warped.lines.by_path

		deskewer = Deskewer(lines)

		polygons = dict()
		for block_path, block in blocks.items():
			polygons[block_path] = deskewer.shapely(
				block.image_space_polygon)

		im = PIL.Image.open(self._page_path)
		self._image = deskewer.image(im)
		self._polygons = polygons


class DewarpedConfiguration(Configuration):
	def __init__(self, page_path):
		super().__init__(page_path)

		reader = Reader([Artifact.CONTOURS], Stage.AGGREGATE, self._page_path)
		blocks = reader.regions.by_path

		polygons = dict()
		for block_path, block in blocks.items():
			polygons[block_path] = block.image_space_polygon

		page = reader.page
		self._image = page.dewarped
		self._polygons = polygons


class OrigamiConfiguration(Configuration):
	def __init__(self, page_path):
		super().__init__(page_path)

		warped = Reader([Artifact.SEGMENTATION], Stage.WARPED, self._page_path)
		reliable = Reader([Artifact.CONTOURS], Stage.RELIABLE, self._page_path)

		separators = Separators(
			warped.segmentation, reliable.separators)
		self._xycut_score = ObstacleSampler(separators)

		contours = dict(
			(k, b.image_space_polygon)
			for k, b in reliable.regions.by_path.items())

		polygons = dict()
		for block_path, polygon in contours.items():
			polygons[block_path] = polygon

		page = reliable.page
		self._image = page.dewarped
		self._polygons = polygons

	@property
	def xycut_score(self):
		return self._xycut_score


class Canvas(QtWidgets.QScrollArea):
	def __init__(self, page_path, configuration):
		super().__init__()

		self.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum,
			QtWidgets.QSizePolicy.Minimum)
		self.setMinimumSize(1280, 600)

		self.label = QtWidgets.QLabel()

		self.setWidgetResizable(True)
		self.setWidget(self.label)

		conf = configuration(page_path)
		im = conf.image
		polygons = conf.polygons
		self._xycut_score = conf.xycut_score
		qt_im = ImageQt(im)

		boxes = []
		for block_path, polygon in polygons.items():
			minx, miny, maxx, maxy = polygon.bounds
			boxes.append(xycut.Box(block_path, minx, miny, maxx, maxy))
		self._boxes = boxes

		self._pixmap = QtGui.QPixmap.fromImage(qt_im)

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
			cut = xycut.XYCut(boxes, score=self._xycut_score)
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
						y0, y1 = cut.extent

						if cut.axis == 0:
							coords = [(cut.x, y0), (cut.x, y1)]
							#print("(%d) x split at %.2f" % (i + 1, cut.x))
						else:
							coords = [(y0, cut.x), (y1, cut.x)]
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
	def __init__(self, page_path, configuration, parent=None):
		super().__init__(parent)
		self.page_path = page_path
		self.setWindowTitle(page_path.name)

		self.canvas = Canvas(page_path, configuration)
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

	path = Path(page_path)
	if path.is_dir() or imghdr.what(path) is None:
		raise click.UsageError("given path needs to point to a page.")

	form = Form(path, OrigamiConfiguration)
	form.show()

	sys.exit(app.exec_())


if __name__ == '__main__':
	app()

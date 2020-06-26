from PySide2 import QtGui, QtCore

import numpy as np
from functools import lru_cache

from origami.core.predict import PredictorType


class Pens:
	def __init__(self, keys, width=10):
		self._pens = dict()

		for i, k in enumerate(keys):
			color = QtGui.QColor.fromHsv(20 + 230 * (i / (1 + len(keys))), 200, 250)
			pen = QtGui.QPen()
			pen.setWidth(width)
			pen.setColor(color)
			pen.setCapStyle(QtCore.Qt.RoundCap)
			self._pens[k] = pen

	def get(self, key):
		return self._pens[key]


def get_region_classes(predictors):
	classes = []
	for p in predictors:
		if p.type == PredictorType.REGION.name:
			for c in p.classes:
				if c != "BACKGROUND":
					classes.append((p.name, c))
	return sorted(classes)


def render_separators(pixmap, separators):
	pens = Pens(sorted(p[:2] for p in separators.keys()))

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.75)

		for line_path, separator in separators.items():
			qp.setPen(pens.get(line_path[:2]))

			pts = [QtCore.QPointF(x, y) for x, y in separator.coords]
			qp.drawPolyline(pts)

	finally:
		qp.end()

	return pixmap


def block_hsv(classes):
	for i, c in enumerate(classes):
		yield tuple(c), (255 * (i / (1 + len(classes))), 100, 200)


class LabelBrushes:
	def __init__(self, predictors):
		self._classes = get_region_classes(predictors)

	@lru_cache(maxsize=8)
	def brushes(self, lighter=0):
		brushes = dict()
		for c, hsv in block_hsv(self._classes):
			brushes[c] = QtGui.QBrush(
				QtGui.QColor.fromHsv(*hsv).lighter(lighter))
		return brushes

	def get_brush(self, block_path, lighter=0):
		classifier, label, block_id = block_path
		return self.brushes(lighter)[(classifier, label)]


def default_pen(color="black", width=5):
	pen = QtGui.QPen()
	pen.setWidth(width)
	pen.setColor(QtGui.QColor(color))
	pen.setCapStyle(QtCore.Qt.RoundCap)
	return pen


def render_blocks(pixmap, blocks, *args, **kwargs):
	contours = dict((k, b.image_space_polygon) for (k, b) in blocks.items())
	return render_contours(pixmap, contours, *args, **kwargs)


def render_contours(pixmap, contours, get_label, predictors=None, brushes=None, matrix=None):
	if brushes is None:
		brushes = LabelBrushes(predictors)

	def point(x, y):
		if matrix is not None:
			x, y = matrix @ np.array([x, y, 1])
		return QtCore.QPointF(x, y)

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.5)

		for block_path, contour in contours.items():
			qp.setBrush(brushes.get_brush(block_path))

			poly = QtGui.QPolygonF()
			for x, y in contour.exterior.coords:
				poly.append(point(x, y))
			qp.drawPolygon(poly)

		qp.setBrush(QtGui.QBrush(QtGui.QColor("white")))

		font = QtGui.QFont("Arial Narrow", 56, QtGui.QFont.Bold)
		qp.setFont(font)

		fm = QtGui.QFontMetrics(font)

		qp.setPen(default_pen())

		for block_path, contour in contours.items():
			x, y = contour.centroid.coords[0]
			p = point(x, y)

			path, label = get_label(block_path)
			qp.setBrush(brushes.get_brush(block_path, lighter=150))

			qp.setOpacity(0.8)
			qp.drawEllipse(p, 50, 50)

			qp.setOpacity(1)
			# flags=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter does
			# not work. fix it manually.
			w = fm.horizontalAdvance(label)
			qp.drawText(p.x() - w / 2, p.y() + fm.descent(), label)

	finally:
		qp.end()

	return pixmap


def render_lines(pixmap, lines, get_label):
	classes = sorted(list(set(x[:2] for x in lines.keys())))
	brushes = dict()
	for c, (h, s, v) in block_hsv(classes):
		brushes[c + (0,)] = QtGui.QBrush(
			QtGui.QColor.fromHsv(h, s, v))
		brushes[c + (1,)] = QtGui.QBrush(
			QtGui.QColor.fromHsv(h, s // 2, v))

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.5)
		qp.setPen(default_pen())

		for i, (line_path, line) in enumerate(lines.items()):
			classifier, label, block_id, line_id = line_path
			qp.setBrush(brushes[(classifier, label, i % 2)])

			poly = QtGui.QPolygonF()
			for x, y in line.image_space_polygon.exterior.coords:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolygon(poly)

			line_info = line.info
			p = np.array(line_info["p"])
			right = np.array(line_info["right"])
			up = np.array(line_info["up"])

			qp.drawPolyline([QtCore.QPointF(*p), QtCore.QPointF(*(p + right))])
			qp.drawPolyline([QtCore.QPointF(*p), QtCore.QPointF(*(p + up))])

	finally:
		qp.end()

	return pixmap


def render_warped_line_paths(pixmap, lines, predictors, resolution=0.1):
	classes = get_region_classes(predictors)
	pens = Pens(classes)

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.9)

		for i, (line_path, line) in enumerate(lines.items()):
			if line.confidence < 0.5:
				continue  # ignore

			classifier, label, block_id, line_id = line_path

			path, height = line.warped_path(resolution)
			pen = pens.get((classifier, label))
			pen.setWidth(int(height / 3))
			qp.setPen(pen)

			poly = QtGui.QPolygonF()
			for x, y in path:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolyline(poly)

	finally:
		qp.end()

	return pixmap


def render_warped_line_confidence(pixmap, lines):
	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		font = QtGui.QFont("Arial Narrow", 48, QtGui.QFont.Bold)
		qp.setFont(font)
		fm = QtGui.QFontMetrics(font)

		for i, (line_path, line) in enumerate(lines.items()):
			if line.confidence < 0.5:
				continue  # ignore

			path, height = line.warped_path(0.1)

			qp.setOpacity(1)
			if line.confidence < 0.75:
				qp.setPen(default_pen("red"))
				label = "%.2f" % line.confidence
				w = fm.horizontalAdvance(label)
				qp.drawText(
					np.mean(path[:, 0]) - w / 2,
					np.mean(path[:, 1]) + fm.descent(), label)

	finally:
		qp.end()

	return pixmap


def render_paths(pixmap, columns):
	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.5)
		qp.setPen(default_pen("blue", 10))

		for path in columns:
			poly = QtGui.QPolygonF()
			for x, y in path:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolyline(poly)

	finally:
		qp.end()

	return pixmap
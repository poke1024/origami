import numpy as np
import networkx as nx
import logging
import collections

from PySide2 import QtGui, QtCore
from functools import lru_cache

from origami.core.predict import PredictorType
from origami.core.neighbors import neighbors


class Pens:
	def __init__(self, keys, width=10):
		self._pens = dict()

		for i, k in enumerate(keys):
			color = QtGui.QColor.fromHsv(
				20 + 230 * (i / (1 + len(keys))), 200, 250)
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

	@lru_cache(maxsize=32)
	def brushes(self, hue=0, saturation=0, value=0, style=QtCore.Qt.SolidPattern):
		brushes = dict()
		for c, (h, s, v) in block_hsv(self._classes):
			brushes[c] = QtGui.QBrush(
				QtGui.QColor.fromHsv(
					(h + hue) % 256, s + saturation, v + value),
				style)
		return brushes

	def get_brush(self, block_path, **kwargs):
		classifier, label, block_id = block_path
		return self.brushes(**kwargs)[(classifier, label)]


def default_pen(color="black", width=5):
	pen = QtGui.QPen()
	pen.setWidth(width)
	pen.setColor(QtGui.QColor(color))
	pen.setCapStyle(QtCore.Qt.RoundCap)
	return pen


def render_blocks(pixmap, blocks, *args, **kwargs):
	contours = dict((k, b.image_space_polygon) for (k, b) in blocks.items())
	return render_contours(pixmap, contours, *args, **kwargs)


_patterns = (
	QtCore.Qt.SolidPattern,
	QtCore.Qt.Dense1Pattern,
	QtCore.Qt.Dense2Pattern,
	QtCore.Qt.Dense3Pattern,
	QtCore.Qt.Dense4Pattern,
	QtCore.Qt.Dense5Pattern
)


def contour_patterns(contours, buffer=-5, threshold=10):
	buffered_contours = dict(
		(k, v.buffer(buffer)) for k, v in contours.items())
	buffered_contours = dict([
		(k, c.convex_hull if c.geom_type != "Polygon" else c)
		for k, c in buffered_contours.items()])
	neighbors_ = neighbors(buffered_contours)
	apart = set()
	for a, b in neighbors_.edges():
		if buffered_contours[a].distance(buffered_contours[b]) > threshold:
			apart.add((a, b))
	for a, b in apart:
		neighbors_.remove_edge(a, b)
	return nx.algorithms.coloring.equitable_color(
		neighbors_, 1 + max(d for _, d in neighbors_.degree()))


def render_contours(
	pixmap, contours, get_label,
	predictors=None, brushes=None, matrix=None,
	alternate=False, edges=None):

	if brushes is None:
		brushes = LabelBrushes(predictors)

	def point(x, y):
		if matrix is not None:
			x, y = matrix @ np.array([x, y, 1])
		return QtCore.QPointF(x, y)

	if alternate:
		patterns = contour_patterns(contours)
	else:
		patterns = None

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.5)

		label_path = collections.defaultdict(list)
		for i, (block_path, contour) in enumerate(contours.items()):
			path, label = get_label(block_path)
			if label is not None:
				label_path[path[:2]].append((label, block_path))

		sorted_contours = dict()
		for k in label_path.keys():
			sorted_contours[k] = [(x[1], contours[x[1]]) for x in sorted(
				label_path[k], key=lambda x: x[0])]

		for k in sorted_contours.keys():
			for i, (block_path, contour) in enumerate(sorted_contours[k]):
				if contour.geom_type != "Polygon":
					logging.error(
						"encountered %s while rendering contour %s" % (
							contour.geom_type, block_path))
					continue

				if patterns:
					style = _patterns[patterns[block_path] % len(_patterns)]
				else:
					style = QtCore.Qt.SolidPattern

				qp.setBrush(brushes.get_brush(
					block_path, style=style))

				poly = QtGui.QPolygonF()
				for x, y in contour.exterior.coords:
					poly.append(point(x, y))
				qp.drawPolygon(poly)

		qp.setBrush(QtGui.QBrush(QtGui.QColor("white")))

		font = QtGui.QFont("Arial Narrow", 56, QtGui.QFont.Bold)
		qp.setFont(font)

		fm = QtGui.QFontMetrics(font)

		qp.setPen(default_pen())
		nodes = dict()
		node_r = 50

		for block_path, contour in contours.items():
			if contour.is_empty:
				continue

			x, y = contour.centroid.coords[0]
			p = point(x, y)

			path, label = get_label(block_path)
			qp.setBrush(brushes.get_brush(block_path, value=50))

			qp.setOpacity(0.8)
			qp.drawEllipse(p, node_r, node_r)
			if edges:
				nodes[block_path] = p

			qp.setOpacity(1)
			# flags=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter does
			# not work. fix it manually.
			label_str = label if isinstance(label, str) else str(label)
			w = fm.horizontalAdvance(label_str)
			qp.drawText(p.x() - w / 2, p.y() + fm.descent(), label_str)

		if edges:
			qp.setOpacity(0.8)
			qp.setPen(default_pen(width=10))

			for p, q in edges:
				coords = [nodes[p], nodes[q]]
				qp.drawPolyline(coords)

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


def render_paths(pixmap, columns, color="blue"):
	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(0.5)
		qp.setPen(default_pen(color, 10))

		for path in columns:
			poly = QtGui.QPolygonF()
			for x, y in path:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolyline(poly)

	finally:
		qp.end()

	return pixmap
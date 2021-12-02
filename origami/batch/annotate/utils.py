#!/usr/bin/env python3

import numpy as np
import networkx as nx
import logging
import collections
import os
import math

try:
	from PySide2 import QtGui, QtCore
except ImportError:
	from PySide6 import QtGui, QtCore

from functools import lru_cache

from origami.core.predict import PredictorType
from origami.core.neighbors import neighbors
from origami.core.math import partition_path


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

			if separator.geom_type != "LineString":
				logging.error(
					"encountered %s while rendering separator %s" % (
						separator.geom_type, line_path))
				continue

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
	pixmap, contours, predictors,
	brushes=None, transform=None, scale=1, font_scale=1,
	get_label=None, alternate=False, edges=None):

	if not contours:
		return pixmap

	if brushes is None:
		brushes = LabelBrushes(predictors)

	def points(pts):
		pts = np.array(pts)
		if transform is not None:
			pts = transform(pts)
		return [QtCore.QPointF(*pt) for pt in (pts * scale)]

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

		def render_contour(coords):
			qp.drawPolygon(points(coords))

		for k in sorted_contours.keys():
			for i, (block_path, contour) in enumerate(sorted_contours[k]):
				if contour.is_empty:
					continue

				if contour.geom_type not in ("Polygon", "MultiPolygon"):
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

				if contour.geom_type == "Polygon":
					render_contour(contour.exterior.coords)
				elif contour.geom_type == "MultiPolygon":
					for geom in contour.geoms:
						render_contour(geom.exterior.coords)
				else:
					raise ValueError(contour.geom_type)

		qp.setBrush(QtGui.QBrush(QtGui.QColor("white")))

		font = QtGui.QFont("Arial Narrow", 56 * scale * font_scale, QtGui.QFont.Bold)
		qp.setFont(font)

		fm = QtGui.QFontMetrics(font)

		qp.setPen(default_pen(width=5 * scale * font_scale))
		nodes = dict()
		node_r = 50 * scale * font_scale

		for block_path, contour in contours.items():
			if contour.is_empty:
				continue

			p = points([contour.centroid.coords[0]])[0]

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
			qp.setPen(default_pen(width=10 * scale))

			for p, q in edges:
				coords = [nodes[p], nodes[q]]
				qp.drawPolyline(coords)

	finally:
		qp.end()

	return pixmap


def render_arrows(qp, path, pos="center", scale=1):
	theta = 45
	d = 25 * scale
	for (x1, y1), (x2, y2) in zip(path, path[1:]):
		dx = x2 - x1
		dy = y2 - y1
		phi = math.atan2(dy, dx)
		phi1 = phi + (90 + theta) * (math.pi / 180)
		phi2 = phi - (90 + theta) * (math.pi / 180)
		if pos == "begin":
			ax, ay = x1, x2
		elif pos == "end":
			ax, ay = x2, y2
		elif pos == "center":
			ax = (x1 + x2) / 2
			ay = (y1 + y2) / 2
		else:
			raise ValueError(pos)
		qp.drawPolyline([
			QtCore.QPointF(ax + d * math.cos(phi1), ay + d * math.sin(phi1)),
			QtCore.QPointF(ax, ay),
			QtCore.QPointF(ax + d * math.cos(phi2), ay + d * math.sin(phi2))
	])


def render_lines(
	pixmap, lines, predictors, scale=1, font_scale=1,
	get_label=None, show_vectors=False):

	if not lines:
		return pixmap

	brushes = LabelBrushes(predictors)

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		black_pen = default_pen(width=5 * scale)
		red_pen = default_pen("#FFA500", width=7 * scale)

		for i, (line_path, line) in enumerate(lines.items()):
			geom_type = line.image_space_polygon.geom_type
			if geom_type != "Polygon":
				logging.error("encountered %s as line geometry" % geom_type)
				continue

			qp.setBrush(brushes.get_brush(line_path[:3], value=(i % 2) * 50))
			qp.setPen(black_pen)

			qp.setOpacity(0.5)
			poly = QtGui.QPolygonF()
			coords = np.array(line.image_space_polygon.exterior.coords) * scale
			for x, y in coords:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolygon(poly)

			if show_vectors:
				p1, p2 = line.baseline
				p1 = np.array(p1) * scale
				p2 = np.array(p2) * scale

				line_info = line.info
				tess_data = line_info["tesseract_data"]

				up = np.array(line_info["up"])
				lh = abs(tess_data["height"]) - abs(tess_data["ascent"])
				up = up * (scale * lh / np.linalg.norm(up))

				qp.setOpacity(0.9)
				qp.setPen(red_pen)
				qp.drawPolyline([QtCore.QPointF(*p1), QtCore.QPointF(*p2)])

				m = (np.array(p1) + np.array(p2)) / 2
				qp.drawPolyline([QtCore.QPointF(*m), QtCore.QPointF(*(m + up))])
				render_arrows(qp, [m, m + up], "end", scale=scale)

		if get_label:
			font = QtGui.QFont("Arial Narrow", 24 * scale * font_scale, QtGui.QFont.Bold)
			qp.setFont(font)
			fm = QtGui.QFontMetrics(font)

			qp.setPen(default_pen(width=5 * scale * font_scale))
			node_r = 25 * scale * font_scale

			for i, (line_path, line) in enumerate(lines.items()):
				x, y = line.image_space_polygon.centroid.coords[0]
				p = QtCore.QPointF(x * scale, y * scale)

				path, label = get_label(line_path)
				qp.setBrush(brushes.get_brush(line_path[:3], value=50))

				qp.setOpacity(0.8)
				qp.drawEllipse(p, node_r, node_r)

				qp.setOpacity(1)
				# flags=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter does
				# not work. fix it manually.
				label_str = label if isinstance(label, str) else str(label)
				w = fm.horizontalAdvance(label_str)
				qp.drawText(p.x() - w / 2, p.y() + fm.descent(), label_str)

	finally:
		qp.end()

	return pixmap


def render_warped_line_paths(pixmap, lines, predictors, resolution=0.1, opacity=0.9):
	classes = get_region_classes(predictors)
	pens = Pens(classes)

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(opacity)

		for i, (line_path, line) in enumerate(lines.items()):
			#if line.confidence < 0.5:
			#	continue  # ignore

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
			#if line.confidence < 0.5:
			#	continue  # ignore

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


def render_paths(
	pixmap, columns,
	color="blue", opacity=0.5,
	show_dir=False, scale=1):

	if not columns:
		return pixmap

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		qp.setOpacity(opacity)
		qp.setPen(default_pen(color, 10))

		for path in columns:
			path = np.array(path) * scale

			poly = QtGui.QPolygonF()
			for x, y in path:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolyline(poly)

			if show_dir:
				for part in partition_path(path, 200):
					render_arrows(qp, part, "center", scale=scale)

	finally:
		qp.end()

	return pixmap

from PySide2 import QtGui, QtCore


class Pens:
	def __init__(self, keys):
		self._pens = dict()

		for i, k in enumerate(keys):
			color = QtGui.QColor.fromHsv(255 * (i / (1 + len(keys))), 200, 200)
			pen = QtGui.QPen()
			pen.setWidth(10)
			pen.setColor(color)
			pen.setCapStyle(QtCore.Qt.RoundCap)
			self._pens[k] = pen

	def get(self, key):
		return self._pens[key]


def render_separators(qt_im, separators):
	pixmap = QtGui.QPixmap.fromImage(qt_im)

	pens = Pens(set(p[:2] for p in separators.keys()))

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

	return pixmap.toImage()


def render_blocks(qt_im, blocks, get_label):
	pixmap = QtGui.QPixmap.fromImage(qt_im)

	pen = QtGui.QPen()
	pen.setWidth(5)
	pen.setColor(QtGui.QColor("black"))
	pen.setCapStyle(QtCore.Qt.RoundCap)

	classes = sorted(list(set(x[:2] for x in blocks.keys())))
	brushes = dict()
	for i, c in enumerate(classes):
		brushes[tuple(c)] = QtGui.QBrush(
			QtGui.QColor.fromHsv(255 * (i / (1 + len(classes))), 100, 200))

	qp = QtGui.QPainter()
	qp.begin(pixmap)

	try:
		font = QtGui.QFont("Arial Narrow", 56, QtGui.QFont.Bold)
		qp.setFont(font)

		fm = QtGui.QFontMetrics(font)

		qp.setPen(pen)
		qp.setOpacity(0.5)

		for block_path, block in blocks.items():
			classifier, label, block_id = block_path
			qp.setBrush(brushes[(classifier, label)])

			poly = QtGui.QPolygonF()
			for x, y in block.image_space_polygon.exterior.coords:
				poly.append(QtCore.QPointF(x, y))
			qp.drawPolygon(poly)

		qp.setBrush(QtGui.QBrush(QtGui.QColor("white")))

		for block_path, block in blocks.items():
			x, y = block.image_space_polygon.centroid.coords[0]

			qp.setOpacity(0.8)
			qp.drawEllipse(QtCore.QPoint(x, y), 50, 50)

			qp.setOpacity(1)
			# flags=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter does
			# not work. fix it manually.
			label = get_label(block_path)
			w = fm.horizontalAdvance(label)
			qp.drawText(x - w / 2, y + fm.descent(), label)

	finally:
		qp.end()

	return pixmap.toImage()

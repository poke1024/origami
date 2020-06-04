import imghdr
import click
import PIL.Image

from PIL.ImageQt import ImageQt
from pathlib import Path
from PySide2 import QtGui, QtCore

from origami.batch.core.block_processor import BlockProcessor


class DebugContoursProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".contours.zip").exists()

	def process(self, p: Path):
		blocks = self.read_blocks(p)

		im = PIL.Image.open(p)
		qt_im = ImageQt(im)
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
				classifier, label, block_id = block_path

				x, y = block.image_space_polygon.centroid.coords[0]

				qp.setOpacity(0.8)
				qp.drawEllipse(QtCore.QPoint(x, y), 50, 50)

				qp.setOpacity(1)
				# flags=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter does
				# not work. fix it.
				label = str(int(block_id))
				w = fm.horizontalAdvance(label)
				qp.drawText(x - w / 2, y + fm.descent(), label)

		finally:
			qp.end()

		pixmap.toImage().save(str(p.with_suffix(".debug.contours.jpg")))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
def debug_contours(data_path, **kwargs):
	""" Export debug information on contours for all document images in DATA_PATH. """
	processor = DebugContoursProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_contours()

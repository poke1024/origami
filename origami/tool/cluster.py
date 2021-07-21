#!/usr/bin/env python3

import sys
import click
import numpy as np
import json
import enum
import io
import zipfile
import sklearn.cluster
import sklearn.neighbors
import random
import matplotlib

from PySide2 import QtWidgets, QtGui, QtCore
from pathlib import Path
from functools import lru_cache
from cached_property import cached_property
from tqdm import tqdm

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.utils import Spinner


class Archive:
	def __init__(self, path, num_labels=3):
		#self._vectors = []
		self._num_labels = num_labels
		self._images = []
		self._path = path
		self._classes = None

		if not path.name.endswith(".zip"):
			raise click.UsageError("path needs to be a zip file.")

		with zipfile.ZipFile(path, "r") as zf:
			for name in tqdm(zf.namelist(), "loading"):
				try:
					if name.startswith("__MACOSX/"):
						continue
					if not name.endswith(".signature.json"):
						continue

					data = json.loads(zf.read(name).decode("utf8"))

					if self._classes is None:
						self._classes = tuple(data["classes"])
					else:
						assert self._classes == tuple(data["classes"])

					data = np.array(data["grid"])
					data = np.clip(data, 0, self._num_labels - 1)

					#self._vectors.append(data.flatten())
					self._images.append(name.rsplit(".", 2)[0])
				except UnicodeDecodeError:
					raise

		#self._vectors = np.array(self._vectors)
		#self._tree = sklearn.neighbors.BallTree(
		#	self._vectors, leaf_size=5, metric="l1")

		#density_k = 5
		#d, i = self._tree.query(
		#	self._vectors, k=density_k, return_distance=True)
		#self._density = np.mean(d[:, 1:], axis=-1)

		self._zf = zipfile.ZipFile(path, "r")

	def close(self):
		self._zf.close()

	@property
	def path(self):
		return self._path

	@property
	def size(self):
		return len(self._images)

	@property
	def classes(self):
		return self._classes

	@property
	def num_labels(self):
		return self._num_labels

	#def vector(self, index):
	#	v = self._vectors[index]
	#	v = v.reshape(9, 9, len(self._classes))  # HACK
	#	return v

	@cached_property
	def aspect_ratio(self):
		r = []
		n = len(self._images)
		for k in random.sample(range(n), min(n, 5)):
			pixmap = self.image(k)
			r.append(pixmap.width() / pixmap.height())
		return np.median(r)

	def image(self, index):
		name = self._images[index] + ".thumbnail.jpg"
		im_bytes = self._zf.read(name)
		pixmap = QtGui.QPixmap()
		pixmap.loadFromData(im_bytes)
		return pixmap

	def name(self, index):
		return self._images[index]

	#def sorted_by_density(self, indices):
	#	densities = self._density[indices]
	#	s = np.argsort(densities)
	#	return indices[s]

	@lru_cache(maxsize=2)
	def cluster(self):
		print("clustering... ", end="", flush=True)
		algorithm = sklearn.cluster.AgglomerativeClustering(
			n_clusters=None, affinity="l1", distance_threshold=25, linkage="single")
		with Spinner():
			clustering = algorithm.fit(self._vectors)
		print("\bdone", flush=True)
		return clustering

	#def neighbor(self, anchor):
	#	d, i = self._tree.query([self._vectors[anchor]], k=2, return_distance=True)
	#	return d[0][-1], i[0][-1]

	#def neighborhood(self, anchor, distance):
	#	return self._tree.query_radius([self._vectors[anchor]], r=distance)[0]


class TableModel(QtCore.QAbstractTableModel):
	COLUMNS = 4
	THUMBNAIL = 256

	def __init__(self, archive, indices=None):
		super(TableModel, self).__init__()

		if indices is None:
			indices = np.arange(archive.size)
		indices = archive.sorted_by_density(indices)

		self._archive = archive
		self._indices = indices

		self._columns = TableModel.COLUMNS
		self._thumnail_size = TableModel.THUMBNAIL

	def page_index(self, row, column):
		index = row * self._columns + column
		if self._indices is not None:
			return self._indices[index]
		else:
			return index

	def page_name(self, index0):
		index = self._indices[index0]
		return self._archive.name(index)

	@lru_cache(maxsize=64)
	def page_image(self, index0):
		index = self._indices[index0]
		pixmap = self._archive.image(index)
		pixmap = pixmap.scaled(QtCore.QSize(
			self._thumnail_size, self._thumnail_size),
			QtCore.Qt.KeepAspectRatio,
			QtCore.Qt.SmoothTransformation)
		return pixmap

	def data(self, index, role):
		if role == QtCore.Qt.ToolTipRole:
			i = index.row() * self._columns + index.column()
			if i < len(self._indices):
				return self.page_name(i).split("/", 1)[-1]

		if role == QtCore.Qt.DecorationRole:
			i = index.row() * self._columns + index.column()
			if i < len(self._indices):
				return self.page_image(i)

	def rowCount(self, index):
		n = len(self._indices)
		c = self._columns
		rows = n // c
		if n % c > 0:
			rows += 1
		return rows

	def columnCount(self, index):
		return self._columns


class NeighborhoodWidget(QtWidgets.QWidget):
	on_exit = QtCore.Signal()

	def __init__(self, archive, anchor, table, parent=None):
		super().__init__(parent)

		self._archive = archive
		self._anchor = anchor
		self._table = table

		self._next_d, _ = archive.neighbor(anchor)

		self.slider = QtWidgets.QSlider()
		self.slider.setOrientation(QtCore.Qt.Horizontal)
		self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
		self.slider.setTickInterval(1)
		self.slider.setMinimum(0)
		self.slider.setMaximum(100)
		self.slider.setValue(10)
		self.slider.valueChanged.connect(self._distance_changed)

		self.label = QtWidgets.QLabel()
		self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

		self._back_button = QtWidgets.QPushButton("back")
		self._back_button.clicked.connect(lambda: self.on_exit.emit())

		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self.slider)
		layout.addWidget(self._back_button)

		layout_v = QtWidgets.QVBoxLayout()
		layout_v.addLayout(layout)
		layout_v.addWidget(self.label)

		self.setLayout(layout_v)

		self._update_model()

	def _update_model(self):
		d = self.slider.value()
		nhood = self._archive.neighborhood(self._anchor, d)
		self.label.setText("d=%d, found %d items, d1=%d." % (
			d, len(nhood), self._next_d))
		self._model = TableModel(
			self._archive,
			nhood)
		self._table.setModel(self._model)

	def _distance_changed(self):
		self._update_model()


class Mode(enum.Enum):
	CLUSTER = 1
	NEIGHBORHOOD = 2


def cluster_name(i):
	if i < 26:
		return chr(ord("A") + i)
	else:
		return cluster_name(i // 26) + cluster_name(i % 26)


class ClusterSelector(QtWidgets.QWidget):
	def __init__(self, table, archive, labels, parent=None):
		super().__init__(parent)

		self._table = table
		self._model = None

		self._archive = archive
		self._labels = labels.copy()

		self._clusters = [None]
		self._combo = QtWidgets.QComboBox()
		self._combo.addItem("all: %d pages" % len(self._labels))

		labels, counts = np.unique(self._labels, return_counts=True)
		cluster_index = 0
		cluttered = 0
		for label, count in zip(labels, counts):
			if count < 5 or label < 0:
				self._labels[self._labels == label] = -1
				cluttered += count
				continue

			self._clusters.append(label)
			self._combo.addItem(
				"cluster %s: %d pages" % (cluster_name(cluster_index), count))
			cluster_index += 1

		if cluttered > 0:
			self._clusters.append(-1)
			self._combo.addItem("clutter: %d pages" % cluttered)

		self._combo.currentIndexChanged.connect(self.switch_to_cluster)
		self.update_table()

		self._details_button = QtWidgets.QPushButton("details")
		self._details_button.setCheckable(True)
		self._details_button.setChecked(False)
		self._details_button.setFlat(True)
		self._details_button.setAutoFillBackground(True)
		self._details_button.setStyleSheet("background-color : lightgrey")
		self._details_button.clicked.connect(self.toggle_details)

		layout = QtWidgets.QHBoxLayout()
		layout.addWidget(self._combo, stretch=5)
		layout.addWidget(self._details_button, stretch=1)
		self.setLayout(layout)

	def toggle_details(self):
		if self._details_button.isChecked():
			self._details_button.setStyleSheet("background-color : lightgreen")
		else:
			self._details_button.setStyleSheet("background-color : lightgrey")

	def update_table(self):
		self.switch_to_cluster(self._combo.currentIndex())

	def switch_to_cluster(self, index):
		label = self._clusters[index]

		if label is None:
			self._model = TableModel(self._archive)
		else:
			indices = np.nonzero(self._labels == label)[0]
			self._model = TableModel(self._archive, indices)

		self._table.setModel(self._model)


class VectorCanvas(QtWidgets.QScrollArea):
	def __init__(self, archive):
		super().__init__()
		self._archive = archive
		self._label = QtWidgets.QLabel()
		self.setWidgetResizable(True)
		self.setWidget(self._label)

	def update_for_page(self, index):
		vector = self._archive.vector(index)
		predictor = 0

		cell_size = 32
		h, w, d = vector.shape
		pixmap = QtGui.QPixmap(
			cell_size * w + 1,
			cell_size * h + 1)
		pixmap.fill()

		cmap = matplotlib.cm.get_cmap('Pastel1')

		palette = [QtGui.QColor.fromRgb(*(np.array(cmap(x)) * 255)) for x in np.linspace(
			0, 1, self._archive.num_labels)]
		brushes = [QtGui.QBrush(x) for x in palette]

		qp = QtGui.QPainter()
		qp.begin(pixmap)

		try:
			pen = QtGui.QPen()
			pen.setWidth(1)
			pen.setColor(QtGui.QColor("black"))
			pen.setCapStyle(QtCore.Qt.RoundCap)
			qp.setPen(pen)

			for ix in range(0, w):
				for iy in range(0, h):
					label = vector[iy, ix, predictor]
					qp.fillRect(
						ix * cell_size, iy * cell_size,
						cell_size, cell_size,
						brushes[label])

			for ix in range(0, w + 1):
				x = ix * cell_size
				qp.drawPolyline([
					QtCore.QPointF(x, 0),
					QtCore.QPointF(x, h * cell_size)])
			for iy in range(0, h + 1):
				y = iy * cell_size
				qp.drawPolyline([
					QtCore.QPointF(0, y),
					QtCore.QPointF(w * cell_size, y)])

		finally:
			qp.end()

		self._label.setPixmap(pixmap)

		self._label.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum,
			QtWidgets.QSizePolicy.Minimum)
		self._label.setMinimumSize(
			pixmap.width(), pixmap.height())


class DetailsPanel(QtWidgets.QWidget):
	def __init__(self, archive, parent=None):
		super().__init__(parent)
		self._archive = archive

		self._combo = QtWidgets.QComboBox()
		for c in self._archive.classes:
			self._combo.addItem(c)

		self._canvas = VectorCanvas(self._archive)

		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self._combo)
		layout.addWidget(self._canvas)
		self.setLayout(layout)

	def update_for_page(self, page):
		self._canvas.update_for_page(page)


class Form(QtWidgets.QDialog):
	def __init__(self, archive, parent=None):
		super().__init__(parent)
		self._archive = archive
		self.setWindowTitle(archive.path.name)

		self.table = QtWidgets.QTableView()
		self.model = None
		self.table.doubleClicked.connect(self.switch_to_neighborhood_view)
		self._mode = Mode.CLUSTER

		#labels = self._archive.cluster().labels_
		labels = np.zeros((self._archive.size, ), dtype=np.uint8)
		self._csel = ClusterSelector(self.table, self._archive, labels)

		self.table.setSizePolicy(
			QtWidgets.QSizePolicy.Minimum,
			QtWidgets.QSizePolicy.Minimum)

		ratio = self._archive.aspect_ratio

		horizontal_header = self.table.horizontalHeader()
		horizontal_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
		horizontal_header.setDefaultSectionSize(TableModel.THUMBNAIL * ratio)
		horizontal_header.hide()

		vertical_header = self.table.verticalHeader()
		vertical_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
		vertical_header.setDefaultSectionSize(TableModel.THUMBNAIL)
		vertical_header.hide()

		min_width = TableModel.THUMBNAIL * ratio * TableModel.COLUMNS
		self.table.setMinimumSize(min(min_width, 1280), 600)

		self._details_panel = DetailsPanel(self._archive)
		#self.table.model.selectionChanged.connect(self.update_details_view)

		table_layout = QtWidgets.QHBoxLayout()
		table_layout.addWidget(self.table)
		table_layout.addWidget(self._details_panel)

		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self._csel)
		layout.addLayout(table_layout)

		self.setLayout(layout)
		self.layout = layout

	def update_details_view(self, index):
		row = index.row()
		column = index.column()
		page = self.table.model().page_index(row, column)
		self._details_panel.update_for_page(page)

	def switch_to_cluster_view(self):
		if self._mode == Mode.CLUSTER:
			return
		self._mode = Mode.CLUSTER

		item = self.layout.takeAt(0)
		item.widget().deleteLater()

		self._csel.show()

		self.model = None
		self.table.doubleClicked.connect(self.switch_to_neighborhood_view)
		self._csel.update_table()

	def switch_to_neighborhood_view(self, index):
		if self._mode == Mode.NEIGHBORHOOD:
			return
		self._mode = Mode.NEIGHBORHOOD

		self.table.doubleClicked.disconnect()
		self._csel.hide()

		row = index.row()
		column = index.column()
		anchor = self.table.model().page_index(row, column)

		widget = NeighborhoodWidget(
			self._archive, anchor, self.table)
		self.layout.insertWidget(0, widget)

		widget.on_exit.connect(self.switch_to_cluster_view)


@click.command()
@click.argument(
	'archive_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def app(archive_path, **kwargs):
	""" Cluster pages in given archive. """
	app = QtWidgets.QApplication(sys.argv)

	archive = Archive(Path(archive_path))

	try:
		form = Form(archive)
		form.show()

		ret = app.exec_()
	finally:
		archive.close()

	sys.exit(ret)


if __name__ == "__main__":
	app()

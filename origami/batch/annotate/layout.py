import imghdr
import click
import PIL.Image
import json
import math
import cv2
import collections
import shapely.ops
import shapely.geometry
import numpy as np

from pathlib import Path
from PySide2 import QtGui
from PIL.ImageQt import ImageQt

from origami.batch.core.block_processor import BlockProcessor
from origami.core.page import Page
from origami.batch.annotate.utils import render_contours, render_paths
from origami.batch.core.lines import reliable_contours


def linestrings(geom):
	if geom is None:
		return
	if geom.geom_type == 'LineString':
		yield geom
	elif geom.geom_type in ('MultiLineString', 'GeometryCollection'):
		for g in geom.geoms:
			for ls in linestrings(g):
				yield ls


def column_paths(shape, x):
	_, miny, _, maxy = shape.bounds
	line = shapely.geometry.LineString([[x, miny - 1], [x, maxy + 1]])
	for geom in linestrings(line.intersection(shape)):
		yield list(geom.coords)


class DebugLayoutProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._overwrite = self._options["overwrite"]

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".aggregate.contours.zip").exists() and\
			p.with_suffix(".aggregate.lines.zip").exists() and\
			p.with_suffix(".dewarped.transform.zip").exists() and\
			p.with_suffix(".order.json").exists() and (
				self._overwrite or
				not p.with_suffix(".annotate.layout.jpg").exists())

	def process(self, page_path: Path):
		contours = self.read_reliable_contours(page_path)

		with open(page_path.with_suffix(".order.json"), "r") as f:
			xycut_data = json.loads(f.read())

		with open(page_path.with_suffix(".tables.json"), "r") as f:
			table_data = json.loads(f.read())

		order = dict(
			(tuple(path.split("/")), i)
			for i, path in enumerate(xycut_data["orders"]["*"]))

		def get_label(block_path):
			return block_path[:2], order[block_path]

		page = Page(page_path, dewarp=True)
		predictors = self.read_predictors(page_path)

		qt_im = ImageQt(page.dewarped)
		pixmap = QtGui.QPixmap.fromImage(qt_im)

		pixmap = render_contours(
			pixmap, contours,
			get_label, predictors, alternate=True)

		columns = []
		for path, xs in table_data["columns"].items():
			path = tuple(path.split("/"))
			for x in xs:
				for coords in column_paths(contours[path], x):
					columns.append(coords)
		pixmap = render_paths(pixmap, columns)

		pixmap.toImage().save(str(
			page_path.with_suffix(".annotate.layout.jpg")))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
@click.option(
	'--overwrite',
	is_flag=True,
	default=False,
	help="Recompute and overwrite existing annotation files.")
def debug_layout(data_path, **kwargs):
	""" Export annotate information on xycuts for all document images in DATA_PATH. """
	processor = DebugLayoutProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	app = QtGui.QGuiApplication()
	debug_layout()

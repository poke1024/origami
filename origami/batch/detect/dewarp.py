#!/usr/bin/env python3

import imghdr
import numpy as np
import json
import click
import shapely.ops
import shapely.wkt
import zipfile
import logging
import PIL.Image

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.dewarp import Dewarper, Grid


def dewarped_contours(input, transformer):
	with open(input.path(Artifact.CONTOURS), "rb") as f:
		with zipfile.ZipFile(f, "r") as zf:
			for name in zf.namelist():
				if name.endswith(".wkt"):
					geom = shapely.wkt.loads(zf.read(name).decode("utf8"))
					geom = shapely.ops.transform(transformer, geom)
					if geom.geom_type not in ("Polygon", "LineString"):
						logging.error("dewarped contour %s is %s" % (
							name, geom.geom_type))
					yield name, geom.wkt.encode("utf8")


class DewarpProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._overwrite = options["overwrite"]

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("input", Input(
				Artifact.CONTOURS,
				Artifact.LINES,
				stage=Stage.WARPED)),
			("output", Output(
				Artifact.DEWARPING_TRANSFORM,
				Artifact.CONTOURS,
				stage=Stage.DEWARPED))
		]

	def process(self, page_path: Path, input, output):
		blocks = input.blocks

		if not blocks:
			return

		lines = input.lines
		separators = input.separators

		page = list(blocks.values())[0].page

		mag = page.magnitude(dewarped=False)
		min_length = mag * self._options["min_line_length"]

		def filter_geoms(geoms, length):
			return dict(
				(k, g) for k, g in geoms.items()
				if length(g) > min_length)

		lines = filter_geoms(lines, lambda l: l.unextended_length)
		separators = filter_geoms(separators, lambda g: g.length)

		grid = Grid.create(
			page,
			blocks, lines, separators,
			grid_res=self._options["grid_cell_size"])

		with output.contours(copy_meta_from=input) as zf:
			for name, data in dewarped_contours(input, grid.transformer):
				zf.writestr(name, data)

		with output.dewarping_transform() as f:
			grid.save(f)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-c', '--grid-cell-size',
	type=int,
	default=25,
	help="grid cell size used for dewarping (smaller is better, but takes longer).")
@click.option(
	'--min-line-length',
	type=float,
	default=0.05,
	help="detect warp using baselines that are above this relative length.")
@click.option(
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
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
	help="Recompute and overwrite existing result files.")
def dewarp(data_path, **kwargs):
	""" Dewarp documents in DATA_PATH. """
	processor = DewarpProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	dewarp()


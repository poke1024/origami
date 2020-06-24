import imghdr
import numpy as np
import json
import click
import shapely.ops
import shapely.wkt
import zipfile
import PIL.Image

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.dewarp import Dewarper, Grid


def dewarped_contours(page_path, transformer):
	with open(page_path.with_suffix(".warped.contours.zip"), "rb") as f:
		with zipfile.ZipFile(f, "r") as zf:
			for name in zf.namelist():
				if name.endswith(".wkt"):
					geom = shapely.wkt.loads(zf.read(name).decode("utf8"))
					yield name, shapely.ops.transform(transformer, geom).wkt.encode("utf8")
				else:
					yield name, zf.read(name)


class DewarpProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".segment.zip").exists() and\
			p.with_suffix(".warped.contours.zip").exists() and\
			p.with_suffix(".warped.lines.zip").exists() and\
			not p.with_suffix(".dewarped.transform.zip").exists() and\
			not p.with_suffix(".dewarped.contours.zip").exists()

	def process(self, page_path: Path):
		separators = self.read_separators(page_path)
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)

		if not blocks:
			return

		page = list(blocks.values())[0].page

		mag = page.magnitude(dewarped=False)
		min_length = mag * self._options["min_line_length"]

		def is_good_line(line):
			return line.unextended_length > min_length

		lines = dict((k, l) for k, l in lines.items() if is_good_line(l))

		grid = Grid.create(
			page.warped.size,
			blocks, lines, separators,
			grid_res=self._options["grid_cell_size"])

		zf_path = page_path.with_suffix(".dewarped.contours.zip")
		with atomic_write(zf_path, mode="wb", overwrite=False) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				for name, data in dewarped_contours(page_path, grid.transformer):
					zf.writestr(name, data)

		out_path = page_path.with_suffix(".dewarped.transform.zip")
		with atomic_write(out_path, mode="wb", overwrite=False) as f:
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
def dewarp(data_path, **kwargs):
	""" Dewarp documents in DATA_PATH. """
	processor = DewarpProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	dewarp()


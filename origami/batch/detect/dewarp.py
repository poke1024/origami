import imghdr
import numpy as np
import json
import click
import shapely.ops
import shapely.wkt
import zipfile

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.core.dewarp import Dewarper
from origami.api import Segmentation


def dewarped_contours(page_path, transformer):
	with open(page_path.with_suffix(".contours.zip"), "rb") as f:
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
		self._dewarp_threshold = options["threshold"]

	def should_process(self, p: Path) -> bool:
		return not p.name.endswith(".dewarped.jpg") and\
			imghdr.what(p) is not None and\
			p.with_suffix(".segment.zip").exists() and\
			p.with_suffix(".contours.zip").exists() and\
			not p.with_suffix(".dewarp.png").exists() and\
			not p.with_suffix(".dewarp.json").exists()

	def process(self, page_path: Path):
		dewarper = Dewarper(page_path, grid_res=self._options["grid_cell_size"])

		# dewarping is expensive and might go wrong. do not do it  unless it seems necessary.
		if dewarper.warping >= self._dewarp_threshold:
			zf_path = page_path.with_suffix(".dewarp.contours.zip")
			with atomic_write(zf_path, mode="wb", overwrite=False) as f:
				with zipfile.ZipFile(f, "w", self.compression) as zf:
					for name, data in dewarped_contours(page_path, dewarper.transformer):
						zf.writestr(name, data)

			im = dewarper.dewarped
			im.save(page_path.with_suffix(".dewarp.png"))

		dewarp_data = dict(warping=dewarper.warping)

		with atomic_write(page_path.with_suffix(".dewarp.json"), mode="w", overwrite=False) as f:
			f.write(json.dumps(dewarp_data))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-t', '--threshold',
	type=float,
	default=0.02,
	help="only dewarp if page is warped above this threshold (0 = always dewarp).")
@click.option(
	'-g', '--grid-cell-size',
	type=int,
	default=25,
	help="grid cell size used for dewarping (smaller is better, but takes longer).")
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


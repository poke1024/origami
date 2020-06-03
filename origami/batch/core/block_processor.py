import zipfile
import json
import shapely.wkt

from pathlib import Path

from origami.batch.core.processor import Processor

from origami.core.page import Page
from origami.core.predict import PredictorType
from origami.core.block import Block, Line


class BlockProcessor(Processor):
	def read_blocks(self, page_path: Path):
		page = Page(page_path)
		blocks = dict()

		with self.lock(page_path.with_suffix(".contours.zip"), "rb") as f:
			with zipfile.ZipFile(f, "r") as zf:
				meta = json.loads(zf.read("meta.json"))

				for name in zf.namelist():
					if not name.endswith(".wkt"):
						continue

					stem = name.rsplit('.', 1)[0]
					parts = tuple(stem.split("/"))
					prediction_name = parts[0]

					t = PredictorType[meta[prediction_name]["type"]]
					if t != PredictorType.REGION:
						continue

					blocks[parts] = Block(
						page, shapely.wkt.loads(zf.read(name).decode("utf8")))

			return blocks

	def read_lines(self, page_path: Path, blocks):
		lines = dict()
		with self.lock(page_path.with_suffix(".lines.zip"), "rb") as lf:
			with zipfile.ZipFile(lf, "r") as zf:
				for name in zf.namelist():
					if not name.endswith(".json"):
						raise RuntimeError("illegal file %s in %s." % (
							name, page_path.with_suffix(".lines.zip")))
					stem = name.rsplit('.', 1)[0]
					parts = tuple(stem.split("/"))
					block = blocks[tuple(parts[:3])]
					line_info = json.loads(zf.read(name))
					lines[stem] = Line(block, **line_info)
		return lines

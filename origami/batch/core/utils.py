import zipfile
import shapely.wkt
import json

from pathlib import Path

from origami.core.page import Page
from origami.core.predict import PredictorType
from origami.core.block import Block, Line


def read_contours(page_path: Path, pred_type, open=open):
	with open(page_path.with_suffix(".contours.zip"), "rb") as f:
		with zipfile.ZipFile(f, "r") as zf:
			meta = json.loads(zf.read("meta.json"))

			for name in zf.namelist():
				if not name.endswith(".wkt"):
					continue

				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				prediction_name = parts[0]

				t = PredictorType[meta[prediction_name]["type"]]
				if t != pred_type:
					continue

				yield parts, shapely.wkt.loads(zf.read(name).decode("utf8"))


def read_blocks(page_path: Path, open=open):
	page = Page(page_path)
	blocks = dict()

	for parts, polygon in read_contours(page_path, PredictorType.REGION, open=open):
		blocks[parts] = Block(page, polygon)

	return blocks


def read_separators(page_path: Path, open=open):
	separators = dict()

	for parts, polygon in read_contours(page_path, PredictorType.SEPARATOR, open=open):
		separators[parts] = polygon

	return separators


def read_lines(page_path: Path, blocks, open=open):
	lines = dict()
	with open(page_path.with_suffix(".lines.zip"), "rb") as lf:
		with zipfile.ZipFile(lf, "r") as zf:
			for name in zf.namelist():
				if not name.endswith(".json"):
					raise RuntimeError("illegal file %s in %s." % (
						name, page_path.with_suffix(".lines.zip")))
				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				block = blocks[tuple(parts[:3])]
				line_info = json.loads(zf.read(name))
				lines[parts] = Line(block, **line_info)
	return lines



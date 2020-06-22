import zipfile
import shapely.wkt
import json

from pathlib import Path

from origami.core.page import Page
from origami.core.predict import PredictorType
from origami.core.block import Block, Line


warp_name = {
	False: "warped",
	True: "dewarped"
}


def read_contours(page_path: Path, pred_type, dewarped=False, open=open):
	name = ".%s.contours.zip" % warp_name[dewarped]
	with open(page_path.with_suffix(name), "rb") as f:
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


def read_blocks(page_path: Path, dewarped=False, open=open):
	page = Page(page_path, dewarped)
	blocks = dict()

	for parts, polygon in read_contours(page_path, PredictorType.REGION, dewarped=dewarped, open=open):
		blocks[parts] = Block(page, polygon, dewarped)

	return blocks


def read_separators(page_path: Path, open=open):
	separators = dict()

	for parts, polygon in read_contours(page_path, PredictorType.SEPARATOR, open=open):
		separators[parts] = polygon

	return separators


def read_lines(page_path: Path, blocks, dewarped=False, open=open):
	assert all(block.dewarped == dewarped for block in blocks.values())
	name = ".%s.lines.zip" % warp_name[dewarped]
	lines = dict()
	with open(page_path.with_suffix(name), "rb") as lf:
		with zipfile.ZipFile(lf, "r") as zf:
			for name in zf.namelist():
				if name == "meta.json":
					continue
				if not name.endswith(".json"):
					raise RuntimeError("illegal file %s in %s." % (
						name, page_path.with_suffix(".lines.zip")))
				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				block = blocks[tuple(parts[:3])]
				line_info = json.loads(zf.read(name))
				lines[parts] = Line(block, **line_info)
	return lines

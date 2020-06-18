import imghdr
import click
import zipfile
import PIL.Image
import json

from pathlib import Path

from origami.batch.core.block_processor import BlockProcessor
from origami.pagexml import pagexml


def parse_line_path(path):
	path = path.rsplit(".", 1)[0]
	region, kind, block_id, line_id = path.split("/")
	return region, kind, int(block_id), int(line_id)


def normalize_text(text):
	text = text.replace("‚‚", "„")
	text = text.replace("''", "\"")
	return text


class DinglehopperProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._use_xy_cut = True

		if options["filter"]:
			self._block_filter = [tuple(options["filter"].split("."))]
		else:
			self._block_filter = None

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".ocr.zip").exists() and\
			p.with_suffix(".xycut.json").exists() and\
			not p.with_suffix(".dinglehopper.xml").exists()

	def process(self, page_path: Path):
		texts = dict()
		with zipfile.ZipFile(page_path.with_suffix(".ocr.zip"), "r") as zf:
			for name in zf.namelist():
				texts[name] = normalize_text(zf.read(name).decode("utf8"))

		paths = list(map(parse_line_path, list(texts.keys())))
		path_to_name = dict(zip(paths, texts.keys()))

		block_paths = sorted(list(set([p[:3] for p in paths])))
		lines = dict((k, []) for k in block_paths)
		for p in paths:
			lines[p[:3]].append(p[3:])

		im = PIL.Image.open(page_path)
		doc = pagexml.Document(page_path.name, im.size)

		if self._use_xy_cut:
			with open(page_path.with_suffix(".xycut.json"), "r") as f:
				xycut_data = json.loads(f.read())

			ordered_blocks = []
			for block_name in xycut_data["order"]:
				region, kind, block_id = block_name.split("/")
				ordered_blocks.append((region, kind, int(block_id)))
		else:
			ordered_blocks = block_paths

		for block_path in ordered_blocks:
			if block_path not in lines:
				continue

			if self._block_filter and block_path[:2] not in self._block_filter:
				continue

			region = pagexml.TextRegion("-".join(map(str, block_path)))
			doc.append(region)

			line_text = []
			for line_path in sorted(lines[block_path]):
				line_text.append(
					texts[path_to_name[block_path + line_path]])

			region.append_text_equiv("\n".join(line_text))

		doc.write(page_path.with_suffix(".dinglehopper.xml"), validate=False)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-f', '--filter',
	type=str,
	default=None,
	help="Only export text from given block path, e.g. -f \"regions.TEXT\".")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def export_for_dinglehopper(data_path, **kwargs):
	""" Export PageXML for use in Dinglehopper for all document images in DATA_PATH. """
	processor = DinglehopperProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_for_dinglehopper()

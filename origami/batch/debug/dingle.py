import imghdr
import click
import zipfile
import PIL.Image

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

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".ocr.zip").exists()

	def process(self, page_path: Path):
		texts = dict()
		with zipfile.ZipFile(page_path.with_suffix(".ocr.zip"), "r") as zf:
			for name in zf.namelist():
				texts[name] = normalize_text(zf.read(name).decode("utf8"))

		paths = list(map(parse_line_path, list(texts.keys())))
		path_to_name = dict(zip(paths, texts.keys()))

		blocks = sorted(list(set([p[:3] for p in paths])))
		lines = dict((k, []) for k in blocks)
		for p in paths:
			lines[p[:3]].append(p[3:])

		im = PIL.Image.open(page_path)
		doc = pagexml.Document(page_path.name, im.size)

		for i, block_path in enumerate(blocks):
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

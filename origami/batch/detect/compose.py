import imghdr
import click
import zipfile
import json
import collections
import codecs
import shapely.strtree

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor


class ComposeProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

		if options["filter"]:
			self._block_filter = [tuple(options["filter"].split("."))]

		# see https://stackoverflow.com/questions/4020539/
		# process-escape-sequences-in-a-string-in-python
		self._block_separator = codecs.escape_decode(bytes(
			self._options["block_separator"], "utf-8"))[0].decode("utf-8")

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".lines.zip").exists() and\
			p.with_suffix(".ocr.zip").exists() and\
			p.with_suffix(".xycut.json").exists() and\
			not p.with_suffix(".compose.txt").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)
		with open(page_path.with_suffix(".xycut.json"), "r") as f:
			xycut_data = json.loads(f.read())

		lines_by_block = collections.defaultdict(list)
		for line_path in lines.keys():
			block_path = line_path[:3]
			lines_by_block[block_path].append(line_path)

		block_polys = [block.image_space_polygon for block in blocks.values()]
		tree = shapely.strtree.STRtree(block_polys)

		line_separator = "\n"
		block_separator = self._block_separator

		page_texts = []
		with zipfile.ZipFile(page_path.with_suffix(".ocr.zip"), "r") as zf:
			for block_name in xycut_data["order"]:
				block_path = tuple(block_name.split("/"))
				if block_path[:2] not in self._block_filter:
					continue

				block = blocks[block_path]

				ignore = False
				for p in tree.query(block.image_space_polygon):
					if p is block.image_space_polygon:
						continue
					if p.contains(block.image_space_polygon):
						ignore = True
						break

				if ignore:
					continue

				block_texts = []
				for line_path in sorted(lines_by_block[block_path]):
					line_text = zf.read("/".join(line_path) + ".txt").decode("utf8")
					block_texts.append(line_text)
				page_texts.append(line_separator.join(block_texts).strip())

		with atomic_write(page_path.with_suffix(".compose.txt"), mode="w", overwrite=False) as f:
			f.write(block_separator.join(page_texts))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-b', '--block-separator',
	type=str,
	default="\n\n",
	help="Character sequence used to separate different blocks.")
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
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

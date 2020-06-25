import imghdr
import click
import zipfile
import json
import collections
import codecs
import shapely.strtree

from pathlib import Path
from atomicwrites import atomic_write
from tabulate import tabulate

from origami.batch.core.block_processor import BlockProcessor
from origami.batch.core.lines import reliable_contours
from origami.core.xycut import polygon_order


def sorted_by_keys(x):
	return [x[k] for k in sorted(list(x.keys()))]


class ComposeProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

		if options["filter"]:
			self._block_filter = [tuple(options["filter"].split("."))]
		else:
			self._block_filter = None

		# see https://stackoverflow.com/questions/4020539/
		# process-escape-sequences-in-a-string-in-python
		self._block_separator = codecs.escape_decode(bytes(
			self._options["block_separator"], "utf-8"))[0].decode("utf-8")

	@property
	def processor_name(self):
		return __loader__.name

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".aggregate.lines.zip").exists() and\
			p.with_suffix(".ocr.zip").exists() and\
			p.with_suffix(".xycut.json").exists() and\
			not p.with_suffix(".compose.txt").exists()

	def process(self, page_path: Path):
		blocks = self.read_aggregate_blocks(page_path)
		if not blocks:
			return

		lines = self.read_aggregate_lines(page_path, blocks)

		reliable = reliable_contours(blocks, lines)
		if self._block_filter is not None:
			reliable = dict(
				(k, v) for k, v in reliable.items()
				if k[:2] in self._block_filter)

		page = list(blocks.values())[0].page
		mag = page.magnitude(dewarped=True)
		fringe = self._options["fringe"] * mag
		order = polygon_order(list(reliable.items()), fringe=fringe)

		line_separator = "\n"
		block_separator = self._block_separator

		page_texts = []
		with zipfile.ZipFile(page_path.with_suffix(".ocr.zip"), "r") as zf:
			def read_text(path):
				return zf.read("/".join(path) + ".txt").decode("utf8")

			lines_by_block = collections.defaultdict(list)
			for line_name in zf.namelist():
				line_path = tuple(line_name.rsplit(".", 1)[0].split("/"))
				block_path = line_path[:3]
				line = lines[line_path[:4]]
				if line.confidence > 0.5:
					lines_by_block[block_path].append(line_path)

			for block_path in order:
				block_lines = lines_by_block.get(block_path, [])
				if block_lines:
					block_texts = []

					if len(block_lines[0]) > 4:  # table with columns?
						rows = collections.defaultdict(dict)
						for line_path in block_lines:
							rows[line_path[:4]][line_path[-1]] = read_text(line_path)

						rows = [sorted_by_keys(row) for row in sorted_by_keys(rows)]
						page_texts.append(tabulate(rows, tablefmt="psql"))
					else:	
						for line_path in sorted(block_lines):
							line_text = read_text(line_path)
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
	'--fringe',
	type=float,
	default=0.001)
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
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

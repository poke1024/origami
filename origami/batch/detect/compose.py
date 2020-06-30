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


def sorted_by_keys(x):
	return [x[k] for k in sorted(list(x.keys()))]


class Table:
	def __init__(self):
		self._divisions = set()
		self._rows = collections.defaultdict(set)
		self._columns = set()
		self._texts = collections.defaultdict(list)

	def append_cell_text(self, grid, text):
		division, row, column = tuple(map(int, grid))
		self._divisions.add(division)
		self._rows[division].add(row)
		self._columns.add(column)
		self._texts[(division, row, column)].append(text)

	def to_text(self):
		columns = sorted(list(self._columns))
		table_data = []
		n_rows = []

		divisions = sorted(list(self._divisions))
		for division in divisions:
			rows = sorted(list(self._rows[division]))
			n_rows.append(len(rows))
			for row in rows:
				row_data = []
				for column in columns:
					texts = [s.strip() for s in self._texts.get(
						(division, row, column), [])]
					row_data.append("\n".join(texts))
				table_data.append(row_data)

		if len(columns) == 1:
			return "\n".join(["".join(x) for x in table_data])
		else:
			if len(n_rows) >= 2 and n_rows[0] == 1:
				headers = "firstrow"
			else:
				headers = ()

			return tabulate(
				table_data, tablefmt="psql", headers=headers)


def sortable_path(line_name):
	line_path = tuple(line_name.rsplit(".", 1)[0].split("/"))
	return line_path[:-1] + (int(line_path[-1]),)


class ComposeProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._overwrite = options["overwrite"]

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
			p.with_suffix(".order.json").exists() and (
				self._overwrite or
				not p.with_suffix(".compose.txt").exists())

	def process(self, page_path: Path):
		blocks = self.read_aggregate_blocks(page_path)
		if not blocks:
			return

		lines = self.read_aggregate_lines(page_path, blocks)

		with open(page_path.with_suffix(".order.json"), "r") as f:
			order_data = json.loads(f.read())
		order = order_data["orders"]["*"]

		line_separator = "\n"
		block_separator = self._block_separator

		page_texts = []
		with zipfile.ZipFile(page_path.with_suffix(".ocr.zip"), "r") as zf:
			def read_text(path):
				return zf.read("/".join(map(str, path)) + ".txt").decode("utf8")

			regular = collections.defaultdict(list)
			tables = collections.defaultdict(Table)
			for line_path in sorted(map(sortable_path, zf.namelist())):
				block_path = line_path[:3]

				# if we cut multiple column images from one line, we will see
				# paths here that are not in "lines".
				# block, division, line, column
				#line = lines[tuple(map(str, line_path))]

				table_path = block_path[2].split(".")
				if len(table_path) > 1:
					tables[block_path[:2] + (table_path[0], )].append_cell_text(
						table_path[1:], read_text(line_path))
				else:  #if line.confidence > 0.5:
					regular[block_path].append(
						read_text(line_path))

			texts_by_block = dict()
			for k, texts in regular.items():
				texts_by_block[k] = line_separator.join(texts).strip()
			for k, table in tables.items():
				texts_by_block[k] = table.to_text()

			for path in map(lambda x: tuple(x.split("/")), order):
				block_text = texts_by_block.get(path, [])
				if block_text:
					page_texts.append(block_text)

		out_path = page_path.with_suffix(".compose.txt")
		with atomic_write(out_path, mode="w", overwrite=self._overwrite) as f:
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
@click.option(
	'--overwrite',
	is_flag=True,
	default=False,
	help="Recompute and overwrite existing result files.")
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

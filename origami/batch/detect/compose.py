#!/usr/bin/env python3

import click
import collections
import codecs

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


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


class ComposeProcessor(Processor):
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

	def artifacts(self):
		return [
			("input", Input(
				Artifact.LINES,
				Artifact.OCR,
				Artifact.ORDER,
				stage=Stage.AGGREGATE)),
			("output", Output(Artifact.COMPOSE)),
		]

	def process(self, page_path: Path, input, output):
		blocks = input.blocks
		if not blocks:
			return

		lines = input.lines

		order_data = input.order
		order = order_data["orders"]["*"]

		line_separator = "\n"
		block_separator = self._block_separator

		page_texts = []

		ocr_data = input.ocr

		def to_text_name(path):
			return "/".join(map(str, path)) + ".txt"

		regular = collections.defaultdict(list)
		tables = collections.defaultdict(Table)
		for line_path in sorted(map(sortable_path, ocr_data.keys())):
			block_path = line_path[:3]

			# if we cut multiple column images from one line, we will see
			# paths here that are not in "lines".
			# block, division, line, column
			#line = lines[tuple(map(str, line_path))]

			ocr_text = ocr_data[to_text_name(line_path)]

			table_path = block_path[2].split(".")
			if len(table_path) > 1:
				tables[block_path[:2] + (table_path[0], )].append_cell_text(
					table_path[1:], ocr_text)
			else:  #if line.confidence > 0.5:
				regular[block_path].append(ocr_text)

		texts_by_block = dict()
		for k, texts in regular.items():
			texts_by_block[k] = line_separator.join(texts).strip()
		for k, table in tables.items():
			texts_by_block[k] = table.to_text()

		for path in map(lambda x: tuple(x.split("/")), order):
			block_text = texts_by_block.get(path, [])
			if block_text:
				page_texts.append(block_text)

		with output.compose() as zf:
			zf.writestr("page.txt", block_separator.join(page_texts))


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
@click.option(
	'--profile',
	is_flag=True,
	default=False)
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

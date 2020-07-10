#!/usr/bin/env python3

import click
import collections
import codecs
import logging

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.utils import RegionsFilter


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


class Composition:
	def __init__(self, line_separator, block_separator):
		self._line_separator = line_separator
		self._block_separator = block_separator
		self._texts = []
		self._path = None

	def append_text(self, path, text):
		text = text.strip()
		if not text:
			return
		assert isinstance(path, tuple)
		if self._path is not None:
			if path[:3] != self._path[:3]:
				self._texts.append(self._block_separator)
		self._path = path
		self._texts.append(text + "\n")

	@property
	def text(self):
		return "".join(self._texts)


class ComposeProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

		if options["regions"]:
			self._block_filter = RegionsFilter(options["regions"])
		else:
			self._block_filter = None

		# see https://stackoverflow.com/questions/4020539/
		# process-escape-sequences-in-a-string-in-python
		self._block_separator = codecs.escape_decode(bytes(
			self._options["paragraph"], "utf-8"))[0].decode("utf-8")

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

		ocr_data = input.ocr

		def to_text_name(path):
			return "/".join(map(str, path)) + ".txt"

		non_tables = collections.defaultdict(dict)
		tables = collections.defaultdict(Table)
		for line_path in sorted(map(sortable_path, ocr_data.keys())):
			block_path = line_path[:3]

			ocr_text = ocr_data[to_text_name(line_path)]

			table_path = block_path[2].split(".")
			if len(table_path) > 1:
				tables[block_path[:2] + (table_path[0], )].append_cell_text(
					table_path[1:], ocr_text)
			else:
				str_line_path = tuple(map(str, line_path))
				non_tables[str_line_path[:3]][str_line_path] = ocr_text

		composition = Composition(
			line_separator="\n",
			block_separator=self._block_separator)

		def append_text_for_block(block_path):
			table = tables.get(block_path)
			if table is not None:
				composition.append_text(
					block_path, tables[block_path].to_text())
			else:
				non_table_data = non_tables.get(block_path)
				if non_table_data:
					sorted_lines = sorted(
						list(non_table_data.items()), key=lambda x: x[0])
					for p, text in sorted_lines:
						composition.append_text(p, text)
				else:
					raise RuntimeError("no text found for %s" % block_path)

		for path in map(lambda x: tuple(x.split("/")), order):
			if self._block_filter is not None and not self._block_filter(path):
				continue
			if len(path) == 3:  # is this a block path?
				append_text_for_block(path)
			elif len(path) == 4:  # is this a line path?
				line_texts = non_tables.get(path[:3])
				if line_texts:
					composition.append_text(path, line_texts[path])
			else:
				raise RuntimeError("illegal path %s in reading order" % path)

		with output.compose() as zf:
			zf.writestr("page.txt", composition.text)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--paragraph',
	type=str,
	default="\n\n",
	help="Character sequence used to separate paragraphs.")
@click.option(
	'--regions',
	type=str,
	default=None,
	help="Only export text from given regions path, e.g. -f \"regions/TEXT\".")
@click.option(
	'--fringe',
	type=float,
	default=0.001)
@Processor.options
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

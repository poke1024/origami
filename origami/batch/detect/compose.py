#!/usr/bin/env python3

import click
import collections
import codecs
import logging
import io
import shapely

from pathlib import Path
from tabulate import tabulate

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.utils import RegionsFilter, TableRegionCombinator
import origami.pagexml.pagexml as pagexml


def sorted_by_keys(x):
	return [x[k] for k in sorted(list(x.keys()))]


class TextRegion:
	def __init__(self, block_path, blocks, lines, transform):
		assert len(blocks) == 1
		self._block = blocks[0][1]
		self._lines = lines
		self._line_texts = dict()
		self._block_path = block_path
		self._transform = transform

	def export_plain_text(self, composition):
		sorted_lines = sorted(
			list(self._line_texts.items()), key=lambda x: x[0])
		for p, text in sorted_lines:
			composition.append_text(p, text)

	def export_plain_line_text(self, composition, line_path):
		composition.append_text(
			line_path, self._line_texts[line_path])

	def export_page_xml(self, px_document):
		px_region = px_document.append_region(
			"TextRegion", id_="-".join(self._block_path))
		px_region.append_coords(self._transform(
			self._block.image_space_polygon.exterior.coords))

		for line_path, line in sorted(list(self._lines.items()), key=lambda x: x[0]):
			px_line = px_region.append_text_line(id_="-".join(line_path))
			px_line.append_coords(self._transform(
				line.image_space_polygon.exterior.coords))
			px_line.append_text_equiv(self._line_texts[line_path])

	def add_text(self, line_path, text):
		self._line_texts[line_path] = text


class TableRegion:
	def __init__(self, block_path, blocks, lines, transform):
		self._lines = lines
		self._block_path = block_path
		self._divisions = set()
		self._rows = collections.defaultdict(set)
		self._columns = set()
		self._texts = collections.defaultdict(list)
		self._transform = transform

		self._blocks = dict()
		for path, block in blocks:
			block_id, division, row, column = map(int, path[2].split("."))
			self._blocks[(column, division, row)] = block

	def export_plain_text(self, composition):
		composition.append_text(
			self._block_path, self.to_text())

	def export_page_xml(self, px_document):
		table_id = "-".join(self._block_path)
		px_table_region = px_document.append_region(
			"TableRegion", id_=table_id)

		columns = sorted(list(self._columns))
		divisions = sorted(list(self._divisions))
		column_shapes = []

		for column in columns:
			column_id = "%s.%d" % (table_id, column)
			px_column = px_table_region.append_text_region(id_=column_id)
			division_shapes = []

			for division in divisions:
				division_id = "%s.%d" % (column_id, division)
				px_division = px_column.append_text_region(id_=division_id)
				cell_shapes = []

				rows = sorted(list(self._rows[division]))
				for row in rows:
					block = self._blocks.get((column, division, row))
					if not block:
						continue

					cell_id = "%s.%d" % (division_id, row)
					px_cell = px_division.append_text_region(id_=cell_id)
					px_cell.append_coords(self._transform(
						block.image_space_polygon.exterior.coords))

					texts = self._texts.get((division, row, column), [])
					for line_path, text in texts:
						px_line = px_cell.append_text_line(
							id_="-".join(line_path))
						px_line.append_coords(self._transform(
							self._lines[line_path].image_space_polygon.exterior.coords))
						px_line.append_text_equiv(text)

					cell_shapes.append(block.image_space_polygon)

				if cell_shapes:
					division_shape = shapely.ops.cascaded_union(cell_shapes)
					px_division.prepend_coords(self._transform(
						division_shape.exterior.coords))
					division_shapes.append(division_shape)
				else:
					px_column.remove(px_division)

			if division_shapes:
				column_shape = shapely.ops.cascaded_union(division_shapes)
				px_column.prepend_coords(self._transform(
					column_shape.exterior.coords))
				column_shapes.append(column_shape)
			else:
				px_table_region.remove(px_column)

		if column_shapes:
			shape = shapely.ops.cascaded_union(column_shapes)
			px_table_region.prepend_coords(self._transform(
				shape.exterior.coords))
		else:
			px_document.remove(px_table_region)

	def append_cell_text(self, grid, line_path, text):
		division, row, column = tuple(map(int, grid))
		self._divisions.add(division)
		self._rows[division].add(row)
		self._columns.add(column)
		self._texts[(division, row, column)].append((line_path, text))

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
					texts = [s.strip() for _, s in self._texts.get(
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


class GraphicRegion:
	def __init__(self, block_path, blocks, lines, transform):
		assert len(blocks) == 1
		self._block = blocks[0][1]
		self._lines = lines
		self._block_path = block_path
		self._transform = transform

	def export_page_xml(self, px_document):
		px_region = px_document.append_region(
			"GraphicRegion", id_="-".join(self._block_path))
		px_region.append_coords(self._transform(
			self._block.image_space_polygon.exterior.coords))


class Document:
	def __init__(self, input):
		self._input = input
		self._grid = self.page.dewarper.grid

		combinator = TableRegionCombinator(input.blocks.keys())
		self._mapping = combinator.mapping

		region_lines = collections.defaultdict(list)
		for line_path, line in input.lines.items():
			region_lines[line_path[:3]].append((line_path, line))
		self._region_lines = region_lines

		self._regions = dict()

		for line_path, ocr_text in input.sorted_ocr:
			block_path = line_path[:3]

			table_path = block_path[2].split(".")
			if len(table_path) > 1:
				base_block_path = block_path[:2] + (table_path[0],)

				self._add(TableRegion, base_block_path).append_cell_text(
					table_path[1:], line_path, ocr_text)
			else:
				self._add(TextRegion, block_path).add_text(
					line_path, ocr_text)

		for block_path, block in input.blocks.items():
			if block_path[:2] == ("regions", "ILLUSTRATION"):
				self._add(GraphicRegion, block_path)

	def transform(self, coords):
		w_coords = self._grid.inverse(coords)
		# Page XML is very picky that we do not specify any
		# negative coordinates. we need to clip.
		width, height = self.page.size(False)
		box = shapely.geometry.box(0, 0, width, height)
		poly = shapely.geometry.Polygon(w_coords).intersection(box)
		return poly.exterior.coords

	def _add(self, class_, block_path):
		region = self._regions.get(block_path)
		if region is None:
			blocks = []
			lines = []
			for path in self._mapping[block_path]:
				blocks.append((path, self._input.blocks[path]))
				lines.extend(self._region_lines[path])
			region = class_(
				block_path, blocks, dict(lines), self.transform)
			self._regions[block_path] = region
		assert isinstance(region, class_)
		return region

	def get(self, block_path):
		region = self._regions.get(block_path)
		if region is None:
			raise RuntimeError("no text found for %s" % str(block_path))
		return region

	@property
	def page(self):
		return list(self._input.blocks.values())[0].page

	@property
	def paths(self):
		return sorted(list(self._regions.keys()))

	def transform(self, order):
		return order


class PlainTextComposition:
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
		self._page_xml = False

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
				Artifact.TABLES,
				stage=Stage.AGGREGATE)),
			("output", Output(Artifact.COMPOSE)),
		]

	def export_page_xml(self, page_path, document):
		page = document.page
		px_document = pagexml.Document(
			filename=str(page_path),
			image_size=page.warped.size)

		for path in document.paths:
			region = document.get(path)
			region.export_page_xml(px_document)

		with io.BytesIO() as f:
			px_document.write(f, overwrite=True, validate=True)
			return f.getvalue()

	def process(self, page_path: Path, input, output):
		blocks = input.blocks
		if not blocks:
			return

		order_data = input.order
		order = order_data["orders"]["*"]

		document = Document(input)

		composition = PlainTextComposition(
			line_separator="\n",
			block_separator=self._block_separator)

		for path in map(lambda x: tuple(x.split("/")), order):
			if self._block_filter is not None and not self._block_filter(path):
				continue

			if len(path) == 3:  # is this a block path?
				document.get(path).export_plain_text(composition)
			elif len(path) == 4:  # is this a line path?
				document.get(path[:3]).export_plain_line_text(composition, path)
			else:
				raise RuntimeError("illegal path %s in reading order" % path)

		with output.compose() as zf:
			zf.writestr("page.txt", composition.text)
			if self._page_xml:
				zf.writestr("page.xml", self.export_page_xml(page_path, document))


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

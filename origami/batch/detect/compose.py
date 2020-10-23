#!/usr/bin/env python3

import click
import collections
import codecs
import logging
import io
import shapely

from pathlib import Path
from tabulate import tabulate
from cached_property import cached_property

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.utils import RegionsFilter, TableRegionCombinator
from origami.batch.core.lines import LineRewriter
import origami.pagexml.pagexml as pagexml


def sorted_by_keys(x):
	return [x[k] for k in sorted(list(x.keys()))]


def polygon_union(geoms):
	if not geoms:
		return None
	shape = shapely.ops.cascaded_union(geoms)
	if shape.geom_type != "Polygon":
		shape = shape.convex_hull
	if shape.is_empty or not shape.is_valid:
		return None
	else:
		return shape


def fix_bogus_tabular_path(path):
	if path[:2] == ("regions", "TABULAR") and "." not in path[2]:
		assert len(path) == 3
		return path[0], path[1], path[2] + ".1.1.1"
	else:
		return path


class MergedTextRegion:
	def __init__(self, document, block_path, lines):
		self._block_path = block_path
		self._polygon = polygon_union([
			line.image_space_polygon for _, line in lines])
		self._document = document
		self._transform = document.rewarp
		self._lines = lines

	def export_page_xml(self, px_document, only_regions):
		if self._polygon is None:
			return

		px_region = px_document.append_region(
			"TextRegion", id_="-".join(self._block_path))
		px_region.append_coords(self._transform(
			self._polygon.exterior.coords))

		if only_regions:
			texts = []
			for i, (line_path, line) in enumerate(self._lines):
				texts.append(self._document.get(line_path[:3]).get_line_text(line_path))
			px_region.append_text_equiv("\n".join(texts))
		else:
			for i, (line_path, line) in enumerate(self._lines):
				line_text = self._document.get(line_path[:3]).get_line_text(line_path)
				px_line = px_region.append_text_line(id_="-".join(self._block_path + (str(i),)))
				px_line.append_coords(self._transform(
					line.image_space_polygon.exterior.coords))
				px_line.append_text_equiv(line_text)


class TextRegion:
	def __init__(self, document, block_path):
		blocks, lines = document.blocks_and_lines(block_path)

		assert len(blocks) == 1
		_, block = blocks[0]
		self._polygon = block.image_space_polygon

		self._block_path = block_path

		self._lines = lines
		self._line_texts = dict()

		self._order = []
		self._transform = document.rewarp

	@property
	def polygon(self):
		return self._polygon

	def get_line_text(self, line_path):
		return self._line_texts[line_path]

	def export_plain_text_region(self, composition):
		for p in self._order:
			composition.append_text(p, self._line_texts[p])

	def export_plain_text_line(self, composition, line_path):
		composition.append_text(
			line_path, self._line_texts[line_path])

	def export_page_xml(self, px_document, only_regions):
		px_region = px_document.append_region(
			"TextRegion", id_="-".join(self._block_path))

		px_region.append_coords(self._transform(
			self._polygon.exterior.coords))

		line_paths = []
		for line_path in self._order:
			line = self._lines[line_path]

			if line.image_space_polygon.is_empty:
				if self._line_texts[line_path]:
					raise RuntimeError(
						"line %s has text '%s', confidence %.2f, but empty geometry" % (
							str(line_path), self._line_texts[line_path], line.confidence))
				continue

			line_paths.append((line_path, line))

		if only_regions:
			texts = []
			for line_path, line in line_paths:
				texts.append(self._line_texts[line_path])
			px_region.append_text_equiv("\n".join(texts))
		else:
			for line_path, line in line_paths:
				px_line = px_region.append_text_line(id_="-".join(line_path))
				px_line.append_coords(self._transform(
					line.image_space_polygon.exterior.coords))
				px_line.append_text_equiv(self._line_texts[line_path])

	def add_text(self, line_path, text):
		self._order.append(line_path)
		self._line_texts[line_path] = text


class TableRegion:
	def __init__(self, document, block_path):
		blocks, lines = document.blocks_and_lines(block_path)

		self._lines = lines
		self._block_path = block_path
		self._divisions = set()
		self._rows = collections.defaultdict(set)
		self._columns = set()
		self._texts = collections.defaultdict(list)
		self._transform = document.rewarp
		self._document = document

		self._blocks = dict()
		for path, block in blocks:
			block_id, division, row, column = map(int, path[2].split("."))
			self._blocks[(column, division, row)] = block

		rewritten = dict()
		for k, line, xs in self._document.rewrite_lines(self._lines):
			rewritten[k] = (line, xs)
		self._rewritten = rewritten

	def export_plain_text_region(self, composition):
		composition.append_text(
			self._block_path, self.to_text())

	def _get_cell_shape(self, cell_line_path):
		# we deal with two different formats of line_path here:
		#
		# cell_line_path we get from OCR-ed text, derived from LineExtractor
		# (generated in _column_path) in OCR stage:
		# : predictor, label, (block, division, 1 + line, x_column), 0
		# here, x_column is a new attribute describing the column split for the
		# line (that covers all columns in the case of non-header divisions).

		# line_path stored in self._lines that was detected during baseline
		# detection in LINES stage:
		# : predictor, label, (block, division, row, column), line
		# (format derives from block id generated in subdivide_table_blocks
		# in LAYOUT stage).
		#
		# we use the inverse mapping in "rewritten" to get from one to the other.

		line, xs = self._rewritten[cell_line_path]

		if xs is None:
			x0, x1 = (None, None)
		else:
			x0, x1 = xs

		line_shape = line.image_space_polygon
		if not (x0 is None and x1 is None):
			minx, miny, maxx, maxy = line_shape.bounds
			if x0 is None:
				x0 = minx
			if x1 is None:
				x1 = maxx
			box = shapely.geometry.box(x0, miny, x1, maxy)
			line_shape = box.intersection(line_shape)
			if line_shape.geom_type != "Polygon":
				line_shape = line_shape.convex_hull

		return line_shape

	def export_page_xml(self, px_document, only_regions):
		table_id = "-".join(self._block_path)
		px_table_region = px_document.append_region(
			"TableRegion", id_=table_id)

		columns = sorted(list(self._columns))
		divisions = sorted(list(self._divisions))
		column_shapes = []

		# make sure to look at subdivide_table_blocks
		# in LAYOUT stage to understand this.

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
					cell_id = "%s.%d" % (division_id, row)
					px_cell = px_division.append_text_region(id_=cell_id)

					line_shapes = []
					texts = self._texts.get((division, row, column), [])
					for cell_line_path, text in texts:

						line_shape = self._get_cell_shape(cell_line_path)
						if line_shape.geom_type == "Polygon" and line_shape.area > 1:
							add_cell = True
							line_shapes.append(line_shape)

						elif text.strip():
							# we have no cell geometry but some text. usually this
							# stems from buffering around a point and the text is
							# meaningless. we omit it.
							# FIXME investigate how these cell geometries come to be.

							add_cell = False

							logging.warning(
								"no cell geometry for text '%s' on page %s" % (
									text, self._document.page_path))
						else:
							add_cell = False

						if add_cell:
							px_line = px_cell.append_text_line(
								id_="-".join(cell_line_path))
							if line_shape is not None:
								px_line.append_coords(self._transform(
									line_shape.exterior.coords))
							px_line.append_text_equiv(text)

					if line_shapes:
						cell_shape = polygon_union(line_shapes)
					else:
						cell_shape = None

					if cell_shape is not None:
						px_cell.prepend_coords(self._transform(
							cell_shape.exterior.coords))
						cell_shapes.append(cell_shape)
					else:
						px_division.remove(px_cell)

				division_shape = polygon_union(cell_shapes)
				if division_shape is not None:
					px_division.prepend_coords(self._transform(
						division_shape.exterior.coords))
					division_shapes.append(division_shape)
				else:
					px_column.remove(px_division)

			column_shape = polygon_union(division_shapes)
			if column_shape is not None:
				px_column.prepend_coords(self._transform(
					column_shape.exterior.coords))
				column_shapes.append(column_shape)
			else:
				px_table_region.remove(px_column)

		table_shape = polygon_union(column_shapes)
		if table_shape is not None:
			px_table_region.prepend_coords(self._transform(
				table_shape.exterior.coords))
		else:
			logging.warning("table %s was empty on page %s." % (
				str(self._block_path), self._document.page_path))
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
	def __init__(self, document, block_path):
		blocks, lines = document.blocks_and_lines(block_path)
		assert len(blocks) == 1
		self._block = blocks[0][1]
		self._lines = lines
		self._block_path = block_path
		self._transform = document.rewarp

	def export_page_xml(self, px_document, only_regions):
		px_region = px_document.append_region(
			"GraphicRegion", id_="-".join(self._block_path))
		px_region.append_coords(self._transform(
			self._block.image_space_polygon.exterior.coords))


class Document:
	def __init__(self, page_path, input, block_filter, text_filter):
		self._page_path = page_path
		self._input = input
		self._grid = self.page.dewarper.grid
		self._rewriter = LineRewriter(input.tables)
		self._block_filter = block_filter
		self._text_filter = text_filter

		combinator = TableRegionCombinator(input.regions.by_path.keys())
		self._mapping = combinator.mapping

		region_lines = collections.defaultdict(list)
		for line_path, line in input.lines.by_path.items():
			region_lines[line_path[:3]].append((line_path, line))
		self._region_lines = region_lines

		self._regions = dict()

		# add lines and line texts in correct order.
		for line_path, ocr_text in input.sorted_ocr:
			ocr_text = self._text_filter(ocr_text)

			block_path = fix_bogus_tabular_path(line_path[:3])
			table_path = block_path[2].split(".")

			if len(table_path) > 1:
				assert block_path[:2] == ("regions", "TABULAR")
				base_block_path = block_path[:2] + (table_path[0],)

				self._add(TableRegion, base_block_path).append_cell_text(
					table_path[1:], line_path, ocr_text)
			else:
				assert block_path[:2] == ("regions", "TEXT")
				self._add(TextRegion, block_path).add_text(
					line_path, ocr_text)

		# add graphics regions.
		for block_path, block in input.regions.by_path.items():
			if block_path[:2] == ("regions", "ILLUSTRATION"):
				self._add(GraphicRegion, block_path)

	@property
	def page_path(self):
		return self._page_path

	@property
	def reading_order(self):
		order_data = self._input.order
		paths = list(map(
			lambda x: tuple(x.split("/")), order_data["orders"]["*"]))
		return list(filter(self._block_filter, paths))

	def rewrite_lines(self, lines):
		return self._rewriter(lines)

	def rewarp(self, coords):
		warped_coords = self._grid.inverse(coords)
		# Page XML is very picky about not specifying any
		# negative coordinates. we need to clip.
		width, height = self.page.size(False)
		box = shapely.geometry.box(0, 0, width, height)
		poly = shapely.geometry.Polygon(warped_coords)
		if not poly.is_valid:
			poly = poly.convex_hull
		page_poly = poly.intersection(box)
		if page_poly.is_empty:
			raise RuntimeError(
				"failed to rewarp coords %s as %s outside page" % (
					str(list(coords)),
					poly))
			return None
		else:
			return page_poly.exterior.coords

	def blocks_and_lines(self, block_path):
		blocks = []
		lines = []
		for path in self._mapping[block_path]:
			fixed_path = fix_bogus_tabular_path(path)
			blocks.append((fixed_path, self._input.regions.by_path[path]))
			lines.extend(self._region_lines[path])
		return blocks, dict(lines)

	def _add(self, class_, block_path):
		region = self._regions.get(block_path)
		if region is None:
			region = class_(self, block_path)
			self._regions[block_path] = region
		assert isinstance(region, class_)
		return region

	def get(self, block_path):
		region = self._regions.get(block_path)
		if region is not None:
			return region

		confidences = [
			l.confidence
			for _, l in self._region_lines[block_path]]
		min_confidence = self._input.lines.min_confidence

		if all(c < min_confidence for c in confidences):
			return None
		else:
			raise RuntimeError(
				"no text found for region %s, line confidences are: %s" % (
					str(block_path), ", ".join(["%.2f" % x for x in confidences])))

	@property
	def page(self):
		return self._input.page

	@property
	def lines(self):
		return self._input.lines

	@cached_property
	def paths(self):
		return sorted(list(self._regions.keys()))


class RegionReadingOrder:
	def __init__(self, document):
		self._document = document

		self._ordered_regions = []
		self._regionless_text_lines = []

		region_indices = collections.defaultdict(int)
		for p in document.paths:
			region_indices[p[:2]] = max(region_indices[p[:2]], int(p[2]))
		self._region_indices = region_indices

		for path in document.reading_order:
			self.append(path)
		self.close()

	def _flush_regionless_lines(self):
		if not self._regionless_text_lines:
			return

		base_path = self._regionless_text_lines[0][:2]
		assert all(p[:2] == base_path for p in self._regionless_text_lines)

		region_indices = self._region_indices
		new_region_index = region_indices[base_path] + 1
		region_indices[base_path] = new_region_index

		new_region_path = base_path + (str(new_region_index),)
		lines = self._document.lines.by_path
		merged = MergedTextRegion(
			self._document,
			new_region_path,
			[(p, lines[p]) for p in self._regionless_text_lines])
		self._ordered_regions.append((new_region_path, merged))
		self._regionless_text_lines = []

	def _is_adjacent(self, line_path):
		if not self._regionless_text_lines:
			return False

		# did lines originally belong to the same region?
		if self._regionless_text_lines[-1][:3] != line_path[:3]:
			return False

		lines = self._document.lines.by_path
		l0 = lines[self._regionless_text_lines[-1]]
		l1 = lines[line_path]

		# FIXME
		if l0.image_space_polygon.distance(l1.image_space_polygon) < 5:
			return True

		return True

	def _add_regionless_line(self, line_path):
		if not self._is_adjacent(line_path):
			self._flush_regionless_lines()

		self._regionless_text_lines.append(line_path)

	def append(self, path):
		if len(path) == 3:  # block path?
			self._flush_regionless_lines()
			region = self._document.get(path)
			if region is not None:
				self._ordered_regions.append((path, region))
		elif len(path) > 3:  # line path?
			assert path[:2] == ("regions", "TEXT")
			self._add_regionless_line(path)
		else:
			raise ValueError("illegal region/line path %s" % str(path))

	def close(self):
		self._flush_regionless_lines()

	@property
	def reading_order(self):
		return [x[0] for x in self._ordered_regions]

	@property
	def regions(self):
		return [x[1] for x in self._ordered_regions]


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


class LetterFilter:
	def __init__(self, ignored):
		self._ignored = ignored

	def __call__(self, t):
		return "".join([c for c in t if c not in self._ignored])


class NullFilter:
	def __call__(self, t):
		return t


class ComposeProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._page_xml = options["page_xml"]
		self._only_page_xml_regions = options["only_page_xml_regions"]

		if options["regions"]:
			self._block_filter = RegionsFilter(options["regions"])
		else:
			self._block_filter = None

		if options["ignore_letters"]:
			ignored = set(options["ignore_letters"])
			self._text_filter = LetterFilter(ignored)
		else:
			self._text_filter = NullFilter()

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
				Artifact.CONTOURS,
				Artifact.LINES,
				Artifact.OCR,
				Artifact.ORDER,
				Artifact.TABLES,
				stage=Stage.RELIABLE)),
			("output", Output(Artifact.COMPOSE)),
		]

	def export_page_xml(self, page_path, document):
		page = document.page

		px_document = pagexml.Document(
			filename=str(page_path),
			image_size=page.warped.size)

		# Page XML does not allow reading orders that
		# contain of regions and lines. We therefore
		# need to merge all line items occurring as
		# separate entities in our reading order into
		# new regions. RegionReadingOrder does that.

		ro = RegionReadingOrder(document)

		px_ro = px_document.append_reading_order()
		px_ro_group = px_ro.append_ordered_group(
			id_="ro_regions", caption="regions reading order")
		for i, path in enumerate(ro.reading_order):
			px_ro_group.append_region_ref_indexed(
				index=i, region_ref="-".join(path))

		for region in ro.regions:
			region.export_page_xml(
				px_document,
				self._only_page_xml_regions)

		with io.BytesIO() as f:
			px_document.write(f, overwrite=True, validate=True)
			return f.getvalue()

	def export_plain_text(self, document):
		composition = PlainTextComposition(
			line_separator="\n",
			block_separator=self._block_separator)

		for path in document.reading_order:
			if self._block_filter is not None and not self._block_filter(path):
				continue

			if len(path) == 3:  # is this a block path?
				region = document.get(path)
				if region is not None:
					region.export_plain_text_region(composition)
			elif len(path) == 4:  # is this a line path?
				region = document.get(path[:3])
				if region is not None:
					region.export_plain_text_line(composition, path)
			else:
				raise RuntimeError("illegal path %s in reading order" % path)

		return composition.text

	def process(self, page_path: Path, input, output):
		if not input.regions.by_path:
			return

		document = Document(page_path, input, self._block_filter, self._text_filter)

		with output.compose() as zf:
			zf.writestr("page.txt", self.export_plain_text(document))
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
	'--page-xml',
	is_flag=True,
	default=False)
@click.option(
	'--only-page-xml-regions',
	is_flag=True,
	default=False)
@click.option(
	'--ignore-letters',
	type=str,
	default="")
@Processor.options
def compose(data_path, **kwargs):
	""" Produce text composed in a single text file for each page in DATA_PATH. """
	processor = ComposeProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	compose()

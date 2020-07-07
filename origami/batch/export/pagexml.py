#!/usr/bin/env python3

import imghdr
import click
import zipfile
import PIL.Image
import json
import collections
import origami.pagexml.pagexml as pagexml

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.core.block import Line


class ExportPageXMLProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return (imghdr.what(p) is not None) and\
			p.with_suffix(".lines.zip").exists() and\
			not p.with_suffix(".xml").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = collections.defaultdict(list)

		with zipfile.ZipFile(page_path.with_suffix(".lines.zip"), "r") as zf:
			for name in zf.namelist():
				assert name.endswith(".json")
				stem = name.rsplit('.', 1)[0]
				parts = tuple(stem.split("/"))
				block_id = tuple(parts[:3])
				block = blocks[block_id]
				line_info = json.loads(zf.read(name))
				lines[block_id].append(Line(block, **line_info))

		im = PIL.Image.open(page_path)
		doc = pagexml.Document(page_path.name, im.size)

		for block_id, block in blocks.items():
			region = pagexml.TextRegion("/".join(block_id))
			region.append_coords(block.coords)
			doc.append(region)

			for j, line in enumerate(lines[block_id]):
				line_node = pagexml.TextLine("line_%d" % (j + 1))

				line_node.append_coords(line.coords)
				region.append(line_node)

		page_xml_path = page_path.with_suffix(".xml")
		if not page_xml_path.exists():
			doc.write(page_xml_path)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
def export_page_xml(data_path, **kwargs):
	""" Export PageXML for all document images in DATA_PATH. Needs
	information from line_detect batch. """
	processor = ExportPageXMLProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_page_xml()


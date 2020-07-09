#!/usr/bin/env python3

import imghdr
import click
import zipfile
import PIL.Image
import json

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output, DebuggingArtifact
from origami.batch.core.utils import RegionsFilter
from origami.pagexml import pagexml


def parse_line_path(path):
	path = path.rsplit(".", 1)[0]
	region, kind, block_id, line_id = path.split("/")
	return region, kind, tuple(map(int, block_id.split("."))), int(line_id)


def text_region_name(path):
	block_id = ".".join(map(str, path[2]))
	return "-".join(map(str, path[:2] + (block_id,)))


def normalize_text(text):
	text = text.replace("‚‚", "„")
	text = text.replace("''", "\"")
	return text


class DinglehopperProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._use_xy_cut = True

		if options["regions"]:
			self._block_filter = RegionsFilter(options["regions"])
		else:
			self._block_filter = None

	def artifacts(self):
		return [
			("input", Input(Artifact.OCR, Artifact.ORDER, stage=Stage.RELIABLE)),
			("output", Output(DebuggingArtifact("dinglehopper.xml"))),
		]

	def process(self, page_path: Path, input, output):
		texts = input.ocr
		texts = dict((k, normalize_text(v)) for k, v in texts.items())

		paths = list(map(parse_line_path, list(texts.keys())))
		path_to_name = dict(zip(paths, texts.keys()))

		block_paths = sorted(list(set([p[:3] for p in paths])))
		lines = dict((k, []) for k in block_paths)
		for p in paths:
			lines[p[:3]].append(p[3:])

		im = PIL.Image.open(page_path)
		doc = pagexml.Document(page_path.name, im.size)

		if self._use_xy_cut:
			xycut_data = input.order

			orders = xycut_data["orders"]
			if self._block_filter and len(self._block_filter.paths) == 1:
				order = orders["/".join(self._block_filter.paths[0])]
			else:
				order = orders["*"]

			ordered_blocks = []
			for block_name in order:
				region, kind, block_id = block_name.split("/")
				ordered_blocks.append((region, kind, (int(block_id),)))
				# FIXME. to include table data here, would need to expand
				# (int(block_id),) into all forms occuring in ocr output.
		else:
			ordered_blocks = block_paths

		for block_path in ordered_blocks:
			if block_path not in lines:
				continue

			if self._block_filter and not self._block_filter(block_path[:2]):
				continue

			region = pagexml.TextRegion(text_region_name(block_path))
			doc.append(region)

			line_text = []
			for line_path in sorted(lines[block_path]):
				line_text.append(
					texts[path_to_name[block_path + line_path]])

			region.append_text_equiv("\n".join(line_text))

		doc.write(
			output.paths[0],
			validate=False,
			overwrite=self._options["overwrite"])


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--regions',
	type=str,
	default="regions/TEXT",
	help="Only export text from given block path, e.g. -f \"regions/TEXT\".")
@Processor.options
def export_for_dinglehopper(data_path, **kwargs):
	""" Export PageXML for use in Dinglehopper for all document images in DATA_PATH. """
	processor = DinglehopperProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_for_dinglehopper()
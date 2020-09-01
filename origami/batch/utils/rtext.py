#!/usr/bin/env python3

import click

from pathlib import Path
from lxml import etree

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
namespaces = {"PAGE": namespace}


class RegionTextProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._output_path = Path(options["output_path"])
		self._min_length = options["min_length"]

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("data", Input(Artifact.COMPOSE))
		]

	def _export_page_xml(self, page_path, root):
		text_regions = dict((r.get("id"), r) for r in root.findall(
			".//PAGE:TextRegion", namespaces=namespaces))

		ogroup = root.findall(
			".//PAGE:OrderedGroup", namespaces=namespaces)
		if not ogroup:
			return
		assert len(ogroup) == 1

		indexed = ogroup[0].findall(
			".//PAGE:RegionRefIndexed", namespaces=namespaces)

		for x in indexed:
			r = text_regions.get(x.get("regionRef"))
			if r is None:
				continue

			line_texts = []
			for unicode in r.findall(
				".//PAGE:TextLine//PAGE:TextEquiv//PAGE:Unicode",
				namespaces=namespaces):
				if unicode.text:
					line_texts.append(unicode.text)

			if line_texts and sum(map(len, line_texts)) > self._min_length:
				name = page_path.stem + (
					"_%03d" % int(x.get("index")))
				with open(self._output_path / (name + ".txt"), "w") as f:
					f.write("\n".join(line_texts))

	def process(self, page_path: Path, data):
		with data.compose as zf:
			if "page.xml" in zf.namelist():
				page_xml = zf.read("page.xml")
				self._export_page_xml(
					page_path,
					etree.fromstring(page_xml))



@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-o', '--output-path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--min-length',
	type=int,
	default=50,
	required=False)
@Processor.options
def rtext(data_path, **kwargs):
	""" Export regions texts for all document images in DATA_PATH. """
	processor = RegionTextProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	rtext()

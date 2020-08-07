#!/usr/bin/env python3

import click
import shapely
import json
import logging
import subprocess

from pathlib import Path
from lxml import etree

from origami.batch.core.processor import Processor
from origami.batch.core.io import Input, Output, Artifact
from origami.batch.detect.compose import ComposeProcessor
from origami.batch.detect.order import ReadingOrderProcessor


def coords_to_shape(coords):
	pts = []
	for pt in coords.attrib["points"].split():
		x, y = pt.split(",")
		pts.append((float(x), float(y)))
	return shapely.geometry.Polygon(pts)


class DinglehopperProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)

	@property
	def processor_name(self):
		return __loader__.name

	def artifacts(self):
		return [
			("data", Input(Artifact.COMPOSE)),
			("output", Output(Artifact.DINGLEHOPPER))
		]

	def process(self, doc_path: Path, data, output):
		gt_path = doc_path.with_suffix(".gt.page.xml")
		if not gt_path.exists():
			logging.warning("no ground truth found for %s" % doc_path)
			return

		config_path = doc_path.with_suffix(".dinglehopper.json")
		if config_path.exists():
			with open(config_path, "r") as f:
				config = json.loads(f.read())
		else:
			config = dict()

		excluded_boxes = []
		for box in config.get("exclude_boxes", []):
			excluded_boxes.append(shapely.geometry.box(*box))

		with data.compose as zf:
			page_xml = zf.read("page.xml")

		namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
		namespaces = {"PAGE": namespace}

		root = etree.fromstring(page_xml)
		for r in list(root.findall(
			".//PAGE:TextRegion", namespaces=namespaces)):

			coords = r.find("PAGE:Coords", namespaces=namespaces)
			for box in excluded_boxes:
				if box.contains(coords_to_shape(coords)):
					# remove references to this region.
					region_id = r.attrib["id"]
					for ref in list(root.findall(
						".//PAGE:RegionRefIndexed",
						namespaces=namespaces)):
						if ref.attrib["regionRef"].strip() == region_id.strip():
							ref.getparent().remove(ref)

					# now remove region itself.
					r.getparent().remove(r)
					break

		tree = etree.ElementTree(root)
		tree.write(
			str(output.path(Artifact.DINGLEHOPPER)),
			encoding='utf-8',
			xml_declaration=True,
			pretty_print=True)

		subprocess.run([
			"dinglehopper",
			str(gt_path),
			str(output.path(Artifact.DINGLEHOPPER)),
			"evaluation_" + doc_path.stem],
			cwd=doc_path.parent)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.argument(
	'ignore-letters',
	type=str,
	default="[]{}")
@Processor.options
def dinglehopper(data_path, **kwargs):
	""" Create dinglehopper.xml for all document images in DATA_PATH. """

	if not kwargs["overwrite"]:
		raise click.UsageError("need --overwrite mode.")

	print("computing reading order...", flush=True)
	order_options = kwargs.copy()
	order_options["disable_region_splitting"] = True
	order_options["ignore"] = "regions/ILLUSTRATION"
	order_options["fringe"] = 0.0025
	order_options["region_area"] = 0.0025
	order_options["splittable"] = "regions/TEXT"
	order_options["separator_flow_width"] = 2
	processor = ReadingOrderProcessor(order_options)
	processor.traverse(data_path)

	print("generating PAGE XMLs...", flush=True)
	compose_options = kwargs.copy()
	compose_options["paragraph"] = "\n\n"
	compose_options["regions"] = "regions/TEXT"
	compose_options["page_xml"] = True
	compose_options["only_page_xml_regions"] = True

	processor = ComposeProcessor(compose_options)
	processor.traverse(data_path)

	print("running CER evaluations...", flush=True)
	processor = DinglehopperProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	dinglehopper()


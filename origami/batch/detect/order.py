#!/usr/bin/env python3

import click
import collections
import logging
import shapely.ops

from pathlib import Path
from itertools import chain

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.lines import reliable_contours
from origami.batch.core.utils import RegionsFilter, TableRegionCombinator
from origami.core.xycut import polygon_order, bounds_order
from origami.core.separate import Separators, ObstacleSampler


def _is_table_path(path):
	return "." in path[2]


class ReadingOrderProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._ignore = RegionsFilter(options["ignore"])
		self._splittable = RegionsFilter(options["splittable"])
		self._min_confidence = options["min_line_confidence"]
		self._enable_region_splitting = not options["disable_region_splitting"]

	@property
	def processor_name(self):
		return __loader__.name

	def compute_order(self, page, contours, lines, sampler):
		contours = dict(contours)
		fringe = page.geometry(dewarped=True).rel_length(self._options["fringe"])

		order = []
		for group in polygon_order(
			contours.items(), fringe=fringe, score=sampler, mode="grouped"):
			if len(group) <= 1 or not self._enable_region_splitting:
				order.extend(group)
			else:
				items = []

				for g in group:
					if self._splittable(g) and not _is_table_path(g):
						for line_path, line in lines[g]:
							p1, p2 = line.baseline
							minx = min(p1[0], p2[0])
							maxx = max(p1[0], p2[0])
							y = (p1[1] + p2[1]) / 2
							tess_data = line.info["tesseract_data"]
							ascent = abs(tess_data['ascent'])
							descent = abs(tess_data['descent'])
							ratio = 0.5  # reduce height
							items.append((line_path, (
								minx, y - ascent * ratio, maxx, y + descent * ratio)))
					else:
						items.append((g, contours[g].bounds))

				for line_path in bounds_order(items, score=sampler):
					order.append(line_path)

		return order

	def xycut_orders(self, page, contours, lines, separators):
		contours = dict((k, v) for k, v in contours.items() if not v.is_empty)

		by_labels = collections.defaultdict(list)
		for p, contour in contours.items():
			if not self._ignore(p):
				by_labels[p[:2]].append((p, contour))

		by_labels[("*",)] = [
			(k, v) for k, v in contours.items() if not self._ignore(k)]

		reliable_region_lines = collections.defaultdict(list)
		for line_path, line in lines.items():
			if line.confidence > self._min_confidence:
				reliable_region_lines[line_path[:3]].append((line_path, line))

		sampler = ObstacleSampler(separators)

		return dict(
			(p, self.compute_order(page, v, reliable_region_lines, sampler))
			for p, v in by_labels.items())

	def artifacts(self):
		return [
			("warped", Input(Artifact.SEGMENTATION, stage=Stage.WARPED)),
			("dewarped", Input(Artifact.CONTOURS, stage=Stage.DEWARPED)),
			("aggregate", Input(Artifact.CONTOURS, Artifact.LINES, stage=Stage.AGGREGATE)),
			("output", Output(Artifact.ORDER, Artifact.CONTOURS, stage=Stage.RELIABLE)),
		]

	def process(self, page_path: Path, warped, dewarped, aggregate, output):
		blocks = aggregate.blocks
		if not blocks:
			return

		page = list(blocks.values())[0].page

		combinator = Combinator(blocks.keys())
		contours = combinator.contours(dict(
			(k, v.image_space_polygon) for k, v in blocks.items()))
		combined_lines = combinator.lines(aggregate.lines)
		reliable = reliable_contours(
			contours,
			combined_lines,
			min_confidence=self._min_confidence,
			ignore=self._ignore)

		min_area = page.geometry(True).rel_area(self._options["region_area"])
		reliable = dict(
			(k, v)
			for k, v in reliable.items()
			if v.area >= min_area)

		separators = Separators(
			warped.segmentation, dewarped.separators)

		with output.contours(copy_meta_from=aggregate) as zf:
			for k, contour in reliable.items():
				if contour.geom_type != "Polygon" and not contour.is_empty:
					logging.error(
						"reliable contour %s is %s" % (k, contour.geom_type))
				zf.writestr("/".join(k) + ".wkt", contour.wkt)

		orders = self.xycut_orders(page, reliable, aggregate.lines, separators)

		orders = dict(
			("/".join(k), ["/".join(map(str, p)) for p in ps])
			for k, ps in orders.items())

		output.order(dict(
			version=1,
			orders=orders))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--ignore',
	type=str,
	default="regions/ILLUSTRATION")
@click.option(
	'--fringe',
	type=float,
	default=0.001)
@click.option(
	'--region-area',
	type=float,
	default=0.0025,
	help="Ignore regions below this relative size.")
@click.option(
	'--splittable',
	type=str,
	default="regions/TEXT")
@click.option(
	'--disable-region-splitting',
	is_flag=True,
	default=False)
@click.option(
	'--min-line-confidence',
	type=float,
	default=0.5)
@Processor.options
def reading_order(data_path, **kwargs):
	""" Detect reading order on all document images in DATA_PATH. """
	processor = ReadingOrderProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	reading_order()


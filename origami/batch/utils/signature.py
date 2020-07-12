#!/usr/bin/env python3

import click
import numpy as np
import shapely
import shapely.strtree
import collections

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output


class SignatureProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

		self._classes = [
			("regions", "TEXT"),
			("regions", "TABULAR"),
			("regions", "ILLUSTRATION")
		]

	@property
	def processor_name(self):
		return __loader__.name

	def grid(self, blocks):
		num_steps = self._options["grid_size"]

		bounds = []
		for block in blocks.values():
			if not block.image_space_polygon.is_empty:
				bounds.append(block.image_space_polygon.bounds)
		bounds = np.array(bounds)

		grid_x = np.linspace(
			np.min(bounds[:, 0]),
			np.max(bounds[:, 2]),
			num_steps + 1)
		grid_y = np.linspace(
			np.min(bounds[:, 1]),
			np.max(bounds[:, 3]),
			num_steps + 1)

		shapes = []
		for block_path, block in blocks.items():
			shape = block.image_space_polygon
			shape.name = "/".join(block_path[:2])
			shapes.append(shape)

		tree = shapely.strtree.STRtree(shapes)

		counts = collections.defaultdict(int)

		for i, (x0, x1) in enumerate(zip(grid_x, grid_x[1:])):
			for j, (y0, y1) in enumerate(zip(grid_y, grid_y[1:])):
				box = shapely.geometry.box(x0, y0, x1, y1)
				for shape in tree.query(box):
					path = tuple(shape.name.split("/"))
					counts[(i, j, path)] += 1

		classes = self._classes
		num_classes = len(classes)

		thumbnail = np.zeros(
			(num_steps, num_steps, num_classes),
			dtype=np.int32)

		for k, p in enumerate(classes):
			for x in range(num_steps):
				for y in range(num_steps):
					z = counts[(x, y, p)]
					thumbnail[y, x, k] = z

		return thumbnail

	def artifacts(self):
		return [
			("input", Input(Artifact.CONTOURS, stage=Stage.AGGREGATE)),
			("output", Output(Artifact.SIGNATURE))
		]

	def process(self, p: Path, input, output):
		grid = self.grid(input.blocks)
		output.signature(dict(
			version=1,
			classes=["/".join(x) for x in self._classes],
			grid=grid.tolist()))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--grid-size',
	type=int,
	default=9)
@Processor.options
def signatures(data_path, **kwargs):
	""" Compute signatures for all document images in DATA_PATH. """
	processor = SignatureProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	signatures()


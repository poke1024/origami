#!/usr/bin/env python3

import click
import numpy as np
import shapely
import shapely.strtree
import shapely.affinity
import collections
import logging
import cv2
import io
import json
import PIL.Image

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.core.canvas import Canvas
from origami.core.math import to_shapely_matrix


def block_bounds(blocks):
	bounds = []
	for block in blocks:
		if not block.image_space_polygon.is_empty:
			bounds.append(block.image_space_polygon.bounds)
	bounds = np.array(bounds)

	return (
		np.min(bounds[:, 0]),
		np.min(bounds[:, 1]),
		np.max(bounds[:, 2]),
		np.max(bounds[:, 3]))


class RegionSignature:
	def __init__(self, regions):
		self._page_bounds = block_bounds(regions)
		minx, miny, maxx, maxy = self._page_bounds
		self._total_area = (maxx - minx) * (maxy - miny)

	def _cell(self, i, j, region_bounds):
		pminx, pminy, pmaxx, pmaxy = self._page_bounds
		rminx, rminy, rmaxx, rmaxy = region_bounds

		xs = (pminx, rminx, rmaxx, pmaxx)
		ys = (pminy, rminy, rmaxy, pmaxy)

		assert -1 <= i <= 1
		assert -1 <= j <= 1
		i1 = i + 1
		j1 = j + 1

		return xs[i1], ys[j1], xs[i1 + 1], ys[j1 + 1]

	def __call__(self, region):
		signature = []

		for i in (-1, 0, 1):
			for j in (-1, 0, 1):
				minx, miny, maxx, maxy = self._cell(i, j, region.bounds)
				area = (maxx - minx) * (maxy - miny)
				signature.append(area / self._total_area)

		return signature


def region_signatures(regions):
	sig = RegionSignature(regions.values())

	for k, region in regions.items():
		sig(region)


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

		minx, miny, maxx, maxy = block_bounds(blocks.values())

		grid_x = np.linspace(minx, maxx, num_steps + 1)
		grid_y = np.linspace(miny, maxy, num_steps + 1)

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
		c_size = 128
		c_buffer = 2

		minx, miny, maxx, maxy = block_bounds(
			input.regions.by_path.values())

		matrix = to_shapely_matrix(cv2.getAffineTransform(
			np.array([(minx, miny), (maxx, miny), (maxx, maxy)], dtype=np.float32),
			np.array([(0, 0), (c_size, 0), (c_size, c_size)], dtype=np.float32)
		))

		thumbnails = dict()

		for k, blocks in input.regions.by_predictors.items():
			canvas = Canvas(c_size, c_size)
			canvas.set_color(1, 1, 1)

			for block in blocks:
				shape = shapely.affinity.affine_transform(
					block.image_space_polygon, matrix)
				shape = shape.buffer(-c_buffer)
				if shape.is_empty:
					continue
				elif shape.geom_type == "Polygon":
					canvas.fill_polygon(
						np.array(shape.exterior.coords))
				elif shape.geom_type == "MultiPolygon":
					for polygon in shape.geoms:
						canvas.fill_polygon(
							np.array(polygon.exterior.coords))
				else:
					logging.error(
						"unexpected geom_type %s" % shape.geom_type)

			mask = canvas.channel("R") > 0
			im = PIL.Image.fromarray(mask.astype(np.uint8) * 255).convert("1")

			with io.BytesIO() as f:
				im.save(f, format="PNG")
				thumbnails["/".join(k)] = f.getvalue()

		with output.signature() as zf:
			meta = dict(
				version=1,
				classes=["/".join(x) for x in self._classes])
			zf.writestr(
				"meta.json", json.dumps(meta).encode("utf8"))
			for k, im_data in thumbnails.items():
				zf.writestr(k + ".png", im_data)


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


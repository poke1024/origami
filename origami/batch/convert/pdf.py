#!/usr/bin/env python3

import click
import sys

from pathlib import Path

from origami.batch.core.processor import Processor

try:
	import pdf2image
except ImportError:
	click.echo("This processor needs pdf2image (see https://pypi.org/project/pdf2image/).")
	click.echo("To install, run this:")
	click.echo("  > pip install pdf2image")
	click.echo("  > conda install -c conda-forge poppler")
	sys.exit(1)


class PDFConverter(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def should_process(self, p: Path) -> bool:
		return p.name.endswith(".pdf") and not (p.parent / (p.stem + "_1.png")).exists()

	def process(self, p: Path):
		images = pdf2image.convert_from_path(p, dpi=self._options["dpi"])
		for i, im in enumerate(images):
			im.save(p.parent / (p.stem + ("_%d.png" % (1 + i))))


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-d', '--dpi',
	type=int,
	default=300)
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def convert_pdfs(data_path, **kwargs):
	""" Convert PDFs in data_path to images for further processing. """
	processor = PDFConverter(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	convert_pdfs()






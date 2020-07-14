#!/usr/bin/env python3

import imghdr
import click
import zipfile
import io
import multiprocessing.pool

from pathlib import Path
from atomicwrites import atomic_write
from cached_property import cached_property

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, DebuggingArtifact, Stage, Input, Output
from origami.pagexml.transcriptions import TranscriptionReader
from origami.batch.core.lines import LineExtractor


class LineExtractionProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	@cached_property
	def output(self):
		name = ["images", "lines"]
		if not self._options["do_not_dewarp"]:
			name.append("dewarped")
		elif not self._options["do_not_deskew"]:
			name.append("deskewed")
		if self._options["binarize"]:
			name.append("binarized")
			name.append(int(self._options["binarize_window_size"]))
		name.append("zip")
		return DebuggingArtifact(".".join(name))

	def artifacts(self):
		if self._options["do_not_dewarp"]:
			stage = Stage.WARPED
			artifacts = [Artifact.LINES]
		else:
			stage = Stage.AGGREGATE
			artifacts = [Artifact.LINES, Artifact.TABLES]

		return [
			("input", Input(*artifacts, stage=stage)),
			("output", Output(self.output))
		]

	def process(self, page_path: Path, input, output):
		lines = input.lines.by_path.items
		tables = None if self._options["do_not_dewarp"] else input.tables

		extractor = LineExtractor(
			tables, self._options["line_height"], self._options)
		images = extractor(lines)

		zip_sep = "-" if self._options["flat"] else "/"

		with output.write_zip_file(self.output) as zf:
			for stem, im in images:
				with io.BytesIO() as f:
					im.save(f, format='png', optimize=True)
					data = f.getvalue()

				zf.writestr("%s.png" % zip_sep.join(stem), data)

			if self._options["export_transcriptions"]:
				page_xml_path = page_path.with_suffix(".xml")
				if page_xml_path.exists():
					r = TranscriptionReader(page_xml_path)
					for stem, line in lines.items():
						text = r.get_text(line)
						if text:
							zf.writestr("%s.txt" % zip_sep.join(stem), text)


@click.command()
@Processor.options
@LineExtractor.options
@click.option(
	'-h', '--line-height',
	default=48,
	type=int,
	help='height of line images in pixels')
@click.option(
	'-t', '--export-transcriptions',
	default=False,
	is_flag=True,
	help='also export transcriptions from matching PageXMLs')
@click.option(
	'-f', '--flat',
	default=False,
	is_flag=True,
	help='generate flat hierarchy in zip file')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
def export_lines(data_path, **kwargs):
	""" Export line images from all document images in DATA_PATH. Needs
	information from previous batches. """
	processor = LineExtractionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_lines()

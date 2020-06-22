import imghdr
import click
import zipfile
import io
import multiprocessing.pool

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.pagexml.transcriptions import TranscriptionReader
from origami.core.block import binarize


class LineExtractionProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options

	def out_path(self, p: Path):
		name = ["images", "lines"]
		if not self._options["do_not_dewarp"]:
			name.append("dewarped")
		elif not self._options["do_not_deskew"]:
			name.append("deskewed")
		if self._options["binarize"]:
			name.append("binarized")
			name.append(int(self._options["binarize_window_size"]))
		name.append("zip")

		return p.with_suffix("." + ".".join(name))

	def should_process(self, p: Path) -> bool:
		if self._options["only_pagexml"]:
			page_xml_path = p.with_suffix(".xml")
			if not page_xml_path.exists():
				return False

		if not self._options["overwrite"]:
			if self.out_path(p).exists():
				return False

		return (imghdr.what(p) is not None) and\
			p.with_suffix(".dewarped.lines.zip").exists()

	def _extract_line_image(self, item):
		stem, line = item
		return stem, line.image(
			target_height=self._options["line_height"],
			dewarped=not self._options["do_not_dewarp"],
			deskewed=not self._options["do_not_deskew"],
			binarized=self._options["binarize"],
			window_size=self._options["binarize_window_size"])

	def process(self, page_path: Path):
		if self._options["do_not_dewarp"]:
			blocks = self.read_blocks(page_path)
			lines = self.read_lines(page_path, blocks)
		else:
			blocks = self.read_dewarped_blocks(page_path)
			lines = self.read_dewarped_lines(page_path, blocks)

		pool = multiprocessing.pool.ThreadPool(processes=8)
		images = pool.map(self._extract_line_image, lines.items())

		zip_sep = "-" if self._options["flat"] else "/"

		with atomic_write(self.out_path(page_path), mode="wb", overwrite=False) as f:
			with zipfile.ZipFile(f, "w") as zf:
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
@click.option(
	'-h', '--line-height',
	default=48,
	type=int,
	help='height of line images in pixels')
@click.option(
	'-w', '--do-not-dewarp',
	default=False,
	is_flag=True,
	help='do not dewarp line images')
@click.option(
	'-s', '--do-not-deskew',
	default=False,
	is_flag=True,
	help='do not deskew line images')
@click.option(
	'--binarize',
	default=False,
	is_flag=True,
	help='binarize line images')
@click.option(
	'--binarize-window-size',
	type=int,
	default=0,
	help="binarization window size for Sauvola or 0 for Otsu.")
@click.option(
	'-t', '--export-transcriptions',
	default=False,
	is_flag=True,
	help='also export transcriptions from matching PageXMLs')
@click.option(
	'-p', '--only-pagexml',
	default=False,
	is_flag=True,
	help='only export pages with matching PageXML')
@click.option(
	'-o', '--overwrite',
	default=False,
	is_flag=True,
	help='overwrite line image data from previous runs')
@click.option(
	'-f', '--flat',
	default=False,
	is_flag=True,
	help='generate flat hierarchy in zip file')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def export_lines(data_path, **kwargs):
	""" Export line images from all document images in DATA_PATH. Needs
	information from previous batches. """
	processor = LineExtractionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_lines()


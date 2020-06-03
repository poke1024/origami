import imghdr
import click
import zipfile
import io

from pathlib import Path
from atomicwrites import atomic_write

from origami.batch.core.block_processor import BlockProcessor
from origami.pagexml.transcriptions import TranscriptionReader


class LineExtractionProcessor(BlockProcessor):
	def __init__(self, options):
		self._options = options

	def should_process(self, p: Path) -> bool:
		if self._options["only_pagexml"]:
			page_xml_path = p.with_suffix(".xml")
			if not page_xml_path.exists():
				return False

		if not self._options["overwrite"]:
			if (p.with_suffix(".line-images.zip")).exists():
				return False

		return (imghdr.what(p) is not None) and\
			p.with_suffix(".lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		lines = self.read_lines(page_path, blocks)

		images_path = page_path.with_suffix(".line-images.zip")
		with atomic_write(images_path, mode="wb", overwrite=False) as f:
			with zipfile.ZipFile(f, "w") as zf:
				for stem, line in lines.items():
					im = line.normalized_image(
						target_height=self._options["line_height"],
						deskewed=not self._options["do_not_deskew"],
						binarized=self._options["binarize"])

					with io.BytesIO() as f:
						im.save(f, format='png', optimize=True)
						data = f.getvalue()

					zf.writestr("%s.png" % stem, data)

				if self._options["export_transcriptions"]:
					page_xml_path = page_path.with_suffix(".xml")
					if page_xml_path.exists():
						r = TranscriptionReader(page_xml_path)
						for stem, line in lines.items():
							text = r.get_text(line)
							if text:
								zf.writestr("%s.txt" % stem, text)


@click.command()
@click.option(
	'-h', '--line-height',
	default=48,
	type=int,
	help='height of line images in pixels')
@click.option(
	'-s', '--do-not-deskew',
	default=False,
	is_flag=True,
	help='do not deskew line images')
@click.option(
	'-b', '--binarize',
	default=False,
	is_flag=True,
	help='binarize line images')
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
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
def export_lines(data_path, **kwargs):
	""" Export line images from all document images in DATA_PATH. Needs
	information from line_detect batch. """
	processor = LineExtractionProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	export_lines()


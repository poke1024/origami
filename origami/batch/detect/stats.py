import imghdr
import click

from pathlib import Path

from origami.batch.core.processor import Processor


class StatsProcessor(Processor):
	def __init__(self, options):
		options["nolock"] = True
		super().__init__(options)

		self._list_names = options["list"]
		if self._list_names:
			self._names = []
		else:
			self._names = None

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None

	def process(self, page_path: Path):
		if self._list_names:
			self._names.append(page_path)

	@property
	def names(self):
		return self._names


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-l', '--list',
	is_flag=True,
	default=False,
	help="List found page names.")
@click.option(
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
def stats(data_path, **kwargs):
	""" List stats of pages in DATA_PATH. """
	processor = StatsProcessor(kwargs)
	processor.traverse(data_path)
	if processor.names is not None:
		print("found %d page(s):" % len(processor.names))
		for name in processor.names:
			print(name)


if __name__ == "__main__":
	stats()

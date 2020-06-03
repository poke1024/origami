import click
import sqlite3
import json

from pathlib import Path
from tqdm import tqdm
import numpy as np

from origami.tool.lineload import LineLoader
from origami.tool.schema import Schema
from origami.tool.split import optimal_split, LineClustering


def _make_line_id(page_path, line_path):
	return ("%s.%s" % (page_path, line_path)).replace("/", ".")


class ExportProcessor:
	def __init__(self, data_path, options):
		self._data_path = Path(data_path)
		self._options = options

		self._schema = Schema(self._options["schema"])
		self._output_path = Path(self._options["output_path"])
		if self._output_path.exists():
			if click.confirm("%s already exists. Do you want to add to it?" % self._output_path):
				pass
			else:
				raise click.Abort()

		db_path = options["db_path"]
		if db_path is None:
			db_path = self._data_path / "annotations.db"
		if not db_path.exists():
			raise click.UsageError("%s does not exist." % db_path)
		self._conn = sqlite3.connect('file:%s?mode=ro' % str(db_path), uri=True)

		self._line_loader = LineLoader()

	def _save_sets(self, sets, text_data, clustering):
		for channel in self._schema.channels:
			if self._options["optimize_split"] != "off":
				texts = []

				preset = {True: [], False: []}
				for (page_path, line_path), (training, validation) in sets.items():
					if training and not validation:
						preset[True].append(len(texts))
					elif validation and not training:
						preset[False].append(len(texts))

					texts.append((
						(page_path, line_path),
						text_data[(page_path, line_path, channel.name)]))

				labels = clustering.labels() if clustering else None

				allocation = dict(zip(
					[t[0] for t in texts], optimal_split(
						[t[1] for t in texts], labels=labels, preset=preset)))

				n_train = sum([int(is_train) for is_train in allocation.values()])
				print("training set size is %d (%.1f%%)." % (n_train, 100 * n_train / len(allocation)))
			else:
				allocation = None

			with open(self._output_path / "txt" / channel.name / "sets.json", "w") as f:
				channel_sets = dict(t=[], v=[], tv=[])

				if allocation:
					for (page_path, line_path), is_train in allocation.items():
						line_id = _make_line_id(page_path, line_path)
						channel_sets["t" if is_train else "v"].append(line_id)
				else:
					for (page_path, line_path), (training, validation) in sets.items():
						line_id = _make_line_id(page_path, line_path)
						if training and validation:
							channel_sets["tv"].append(line_id)
						elif training:
							channel_sets["t"].append(line_id)
						elif validation:
							channel_sets["v"].append(line_id)

				f.write(json.dumps(channel_sets))

	def run(self):
		cursor = self._conn.cursor()
		cursor.execute("SELECT page_path, line_path, annotation, training, validation FROM lines")
		line_data = cursor.fetchall()
		cursor.close()

		text_data = dict()
		sets = dict()

		for row in tqdm(line_data, desc="preprocessing"):
			page_path, line_path, annotation, training, validation = row
			try:
				for channel in self._schema.channels:
					text_data[(page_path, line_path, channel.name)] = channel.transform(annotation)
			except:
				click.echo("Error in line %s/%s." % (page_path, line_path))
				raise

			any_text_data = False
			for channel in self._schema.channels:
				any_text_data = any_text_data or len(
					text_data[(page_path, line_path, channel.name)].strip()) > 0
			if any_text_data:
				sets[(page_path, line_path)] = (training, validation)

		image_channel = "%s%s%d" % (
			"skewed-" if self._options["do_not_deskew"] else "deskewed-",
			"bin-" if self._options["binarized"] else "gray-",
			self._options["line_height"]
		)

		self._output_path.mkdir(exist_ok=True)

		(self._output_path / "txt").mkdir(exist_ok=True)
		for channel in self._schema.channels:
			(self._output_path / "txt" / channel.name).mkdir(exist_ok=True)

		(self._output_path / "img").mkdir(exist_ok=True)
		(self._output_path / "img" / image_channel).mkdir(exist_ok=True)

		if self._options["optimize_split"] == "best":
			clustering = LineClustering()
		else:
			clustering = None

		for page_path, line_path in tqdm(sets.keys(), desc="exporting"):
			line_id = _make_line_id(page_path, line_path)

			for channel in self._schema.channels:
				with open(self._output_path / "txt" / channel.name / ("%s.gt.txt" % line_id), "w") as f:
					f.write(text_data[(page_path, line_path, channel.name)])

			im = self._line_loader.load_line_image(
				self._data_path / page_path,
				line_path,
				target_height=self._options["line_height"],
				deskewed=not self._options["do_not_deskew"],
				binarized=self._options["binarized"])

			im.save(self._output_path / "img" / image_channel / ("%s.png" % line_id))

			if clustering:
				clustering.add(im)

		self._save_sets(sets, text_data, clustering)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-o', '--output-path',
	type=click.Path(exists=False),
	required=True,
	help="Where exported files are written.")
@click.option(
	'-s', '--schema',
	type=click.File(),
	required=False,
	help="Schema for output folder contents.")
@click.option(
	'-h', '--line-height',
	type=int,
	default=48,
	help="Pixel height of exported line images.")
@click.option(
	'-b', '--binarized',
	is_flag=True,
	default=False,
	help="Binarize exported line images.")
@click.option(
	'-s', '--do-not-deskew',
	default=False,
	is_flag=True,
	help='do not deskew line images')
@click.option(
	'--optimize-split',
	type=click.Choice(['off', 'fast', 'best'], case_sensitive=False),
	default='off',
	help='optimize training set selection')
@click.option(
	'--db-path',
	type=click.Path(exists=False),
	help="Path to db.")
def export(data_path, **kwargs):
	""" Export annotation database into image and text files. """
	processor = ExportProcessor(data_path, kwargs)
	processor.run()


if __name__ == "__main__":
	export()

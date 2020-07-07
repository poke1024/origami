import imghdr
import click
import random
import collections
import sqlite3
import functools

from pathlib import Path

from origami.batch.core.processor import Processor
from origami.pagexml.transcriptions import TranscriptionReader


def _sample_all(lines):
	return lines


def _sample_n(lines, n):
	return random.sample(lines, min(len(lines), n))


def _parse_samplers(spec):
	if spec.lower().strip() == "all":
		return None

	counts = dict()
	for region_spec in spec.split(","):
		key_value = region_spec.strip().split(":")
		if len(key_value) == 1:
			sampler = _sample_all
			r = key_value[0]
		else:
			r, n_spec = key_value

			if n_spec.lower() == "all":
				sampler = _sample_all
			else:
				n = int(n_spec)
				sampler = functools.partial(_sample_n, n=n)

		counts[tuple(r.split("."))] = sampler

	return counts


class SampleProcessor(Processor):
	def __init__(self, data_path, options):
		super().__init__(options)
		self._data_path = Path(data_path)
		self._options = options

		self._lines = []
		try:
			self._samplers = _parse_samplers(
				self._options["sample"])
		except:
			raise click.BadParameter(
				"expected region:count syntax.", param_hint="sample")

		random.seed(self._options["seed"])

		db_path = options["db_path"]
		if db_path is None:
			db_path = self._data_path / "annotations.db"
		else:
			db_path = Path(db_path)
		#if db_path.exists():
		#	raise click.UsageError("%s already exists." % db_path)
		self._conn = sqlite3.connect(db_path)

		with self._conn:
			self._conn.execute(
				'''CREATE TABLE IF NOT EXISTS lines(page_path TEXT, 
				line_path TEXT, line_wkt TEXT, annotation TEXT, author TEXT,
				training BOOLEAN, validation BOOLEAN)''')

			self._conn.execute(
				"CREATE UNIQUE INDEX IF NOT EXISTS unique_line ON lines(page_path, line_path)")

		cursor = self._conn.cursor()
		cursor.execute("SELECT page_path, line_path FROM lines")
		db_line_data = cursor.fetchall()
		cursor.close()
		self._ignored_lines = collections.defaultdict(set)
		for page_path, line_path in db_line_data:
			self._ignored_lines[page_path].add(line_path)

	def close(self):
		self._conn.close()

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".lines.zip").exists()

	def process(self, page_path: Path):
		blocks = self.read_blocks(page_path)
		all_lines = self.read_lines(page_path, blocks)

		relative_page_path = page_path.relative_to(self._data_path)

		ignored = self._ignored_lines[str(relative_page_path)]

		dup = set(all_lines.keys()) & ignored
		if dup:
			for k in dup:
				del all_lines[k]

		lines_by_region = collections.defaultdict(list)
		for parts, line in all_lines.items():
			line_path = "/".join(parts)
			lines_by_region[tuple(parts[:2])].append(
				(relative_page_path, line_path, line))

		reader = None
		if self._options["import_pagexml"] or self._options["only_transcribed"]:
			page_xml_path = page_path.with_suffix(".xml")
			if page_xml_path.exists():
				reader = TranscriptionReader(page_xml_path)

		samplers = self._samplers
		if not samplers:
			samplers = dict((k, _sample_all) for k in lines_by_region.keys())

		all_region_lines = []
		slices = dict()

		for r, sampler in samplers.items():
			region_lines = lines_by_region[r]
			slices[r] = slice(len(all_region_lines), len(all_region_lines) + len(region_lines))
			all_region_lines.extend(region_lines)

		if reader:
			reader.fetch_texts([line for pp, lp, line in all_region_lines])

		rows = []
		for r, s in slices.items():
			region_lines = all_region_lines[s]

			if region_lines:
				sampled = sampler(region_lines)

				if self._options["only_transcribed"]:
					sampled = [(pp, lp, line) for pp, lp, line in sampled if line.text]

				rows.extend([
					(str(pp), str(lp), line.image_space_polygon.wkt, line.text, True, True)
					for pp, lp, line in sampled])

		with self._conn:
			self._conn.executemany(
				'''INSERT INTO lines(page_path, line_path, line_wkt, annotation, training, validation)
				VALUES (?, ?, ?, ?, ?, ?)''', rows)

		if reader:
			reader.log_unfetched()


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-S', '--seed',
	type=int,
	default=1267985421,
	help="Random seed for filtering.")
@click.option(
	'-s', '--sample',
	type=str,
	default="regions.TEXT:5, regions.TABULAR:1",
	help="Which regions and lines to sample. Use 'all' for all.")
@click.option(
	'--db-path',
	type=click.Path(exists=False),
	help="Path to db.")
@click.option(
	'--import-pagexml',
	default=False,
	is_flag=True,
	help="Import transcription data from PageXML.")
@click.option(
	'-t', '--only-transcribed',
	default=False,
	is_flag=True,
	help="Import only transcribed lines.")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def sample_lines(data_path, **kwargs):
	""" Create line database from document images in DATA_PATH.
	Information from lines batch needs to be present. """
	processor = SampleProcessor(data_path, kwargs)
	processor.traverse(data_path)
	processor.close()


if __name__ == "__main__":
	sample_lines()

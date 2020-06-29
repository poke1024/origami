import imghdr
import click
import zipfile
import numpy as np
import time
import multiprocessing.pool

from pathlib import Path
from functools import partial
from atomicwrites import atomic_write

from calamari_ocr.ocr import Predictor, MultiPredictor
from calamari_ocr.ocr.voting.confidence_voter import ConfidenceVoter

from origami.batch.core.block_processor import BlockProcessor
from origami.core.binarize import Binarizer
from origami.batch.core.lines import LineExtractor


class OCRProcessor(BlockProcessor):
	def __init__(self, options):
		super().__init__(options)
		self._model_path = Path(options["model"])
		self._options = options

		models = list(self._model_path.glob("*.json"))
		models = [m for m in models if m.with_suffix(".h5").exists()]
		if len(models) < 1:
			raise FileNotFoundError(
				"no Calamari models found at %s" % self._model_path)
		self._models = models

		self._predictor = None
		self._voter = None
		self._line_height = None

		# bbz specific.
		self._ignored = set([("regions", "ILLUSTRATION")])

	@property
	def processor_name(self):
		return __loader__.name

	def _load_models(self):
		if self._predictor is not None:
			return

		batch_size = self._options["batch_size"]
		if batch_size > 0:
			batch_size_kwargs = dict(batch_size=batch_size)
		else:
			batch_size_kwargs = dict()

		if len(self._models) == 1:
			self._predictor = Predictor(
				str(self._models[0]), **batch_size_kwargs)
			self._predict_kwargs = batch_size_kwargs
			self._voter = None
			self._line_height = int(self._predictor.model_params.line_height)
		else:
			print("using Calamari voting with %d models." % len(self._models))
			self._predictor = MultiPredictor(
				checkpoints=[str(p) for p in self._models],
				**batch_size_kwargs)
			self._predict_kwargs = dict()
			self._voter = ConfidenceVoter()
			self._line_height = int(self._predictor.predictors[0].model_params.line_height)

	def should_process(self, p: Path) -> bool:
		return imghdr.what(p) is not None and\
			p.with_suffix(".aggregate.lines.zip").exists() and\
			p.with_suffix(".dewarped.transform.zip").exists() and\
			not p.with_suffix(".ocr.zip").exists()

	def process(self, page_path: Path):
		self._load_models()

		blocks = self.read_aggregate_blocks(page_path)
		lines = self.read_aggregate_lines(page_path, blocks)

		extractor = LineExtractor(self._line_height, self._options)

		names = []
		images = []
		for stem, im in extractor(page_path, lines, ignored=self._ignored):
			names.append("/".join(stem))
			images.append(np.array(im))

		texts = []
		for prediction in self._predictor.predict_raw(
			images, progress_bar=False, **self._predict_kwargs):

			if self._voter is not None:
				prediction = self._voter.vote_prediction_result(prediction)
			texts.append(prediction.sentence)

		zf_path = page_path.with_suffix(".ocr.zip")
		with atomic_write(zf_path, mode="wb", overwrite=False) as f:
			with zipfile.ZipFile(f, "w", self.compression) as zf:
				for name, text in zip(names, texts):
					zf.writestr("%s.txt" % name, text)


@click.command()
@click.option(
	'-m', '--model',
	required=True,
	type=click.Path(exists=True),
	help='path that contains Calamari model(s)')
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@click.option(
	'-b', '--batch-size',
	type=int,
	default=-1,
	required=False)
@click.option(
	'--binarize',
	is_flag=True,
	default=False,
	help="binarize line images (important if your model expects this).")
@click.option(
	'--binarize-window-size',
	type=int,
	default=0,
	help="binarization window size for Sauvola or 0 for Otsu.")
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
	'--name',
	type=str,
	default="",
	help="Only process paths that conform to the given pattern.")
@click.option(
	'--nolock',
	is_flag=True,
	default=False,
	help="Do not lock files while processing. Breaks concurrent batches, "
	"but is necessary on some network file systems.")
def segment(data_path, **kwargs):
	""" Perform OCR on all recognized lines in DATA_PATH. """
	processor = OCRProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()


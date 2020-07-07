#!/usr/bin/env python3

import click
import numpy as np

from pathlib import Path

from calamari_ocr.ocr import Predictor, MultiPredictor
from calamari_ocr.ocr.voting.confidence_voter import ConfidenceVoter

from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.lines import LineExtractor
from origami.batch.core.utils import RegionsFilter


class OCRProcessor(Processor):
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

		self._ignored = RegionsFilter(options["ignore"])

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

	def artifacts(self):
		return [
			("aggregate", Input(
				Artifact.LINES, Artifact.TABLES,
				stage=Stage.AGGREGATE)),
			("output", Output(Artifact.OCR)),
		]

	def process(self, page_path: Path, aggregate, output):
		self._load_models()

		lines = aggregate.lines

		extractor = LineExtractor(
			aggregate.tables, self._line_height, self._options)

		names = []
		images = []
		for stem, im in extractor(lines, ignored=self._ignored):
			names.append("/".join(stem))
			images.append(np.array(im))

		texts = []
		for prediction in self._predictor.predict_raw(
			images, progress_bar=False, **self._predict_kwargs):

			if self._voter is not None:
				prediction = self._voter.vote_prediction_result(prediction)
			texts.append(prediction.sentence)

		with output.ocr() as zf:
			for name, text in zip(names, texts):
				zf.writestr("%s.txt" % name, text)


@click.command()
@Processor.options
@LineExtractor.options
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
	'--ignore',
	type=str,
	default="regions/ILLUSTRATION")
def segment(data_path, **kwargs):
	""" Perform OCR on all recognized lines in DATA_PATH. """
	processor = OCRProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	segment()


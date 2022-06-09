#!/usr/bin/env python3

import click
import numpy as np
import logging
import functools
import PIL.Image

from pathlib import Path


class CalamariPredictor:
	def __init__(self, predictor):
		self._predictor = predictor
		
	def __call__(self, images, **kwargs):
		for prediction in self._predictor.predict_raw(
			images, progress_bar=False, **kwargs):
			yield prediction.sentence


class VotedCalamariPredictor:
	def __init__(self, predictor, voter):
		self._predictor = predictor
		self._voter = voter
		
	def __call__(self, images, **kwargs):
		for prediction in self._predictor.predict_raw(
			images, progress_bar=False, **kwargs):
	   		yield self._voter.vote_prediction_result(prediction).sentence


class Calamari:
	def __init__(self):
		import calamari_ocr
		calamari_version = tuple(map(int, calamari_ocr.__version__.split('.')))

		print(f"calamari_ocr version is {calamari_version}.")
		if calamari_version >= (2, 1):
			from calamari_ocr.ocr.predict.predictor import Predictor, MultiPredictor, PredictorParams
		
			def load_predictor(path, **kwargs):
				return CalamariPredictor(Predictor.from_checkpoint(
					params=PredictorParams(),
					checkpoint=path))
		
			def load_multi_predictor(paths, **kwargs):
				return CalamariPredictor(MultiPredictor.from_paths(
					checkpoints=paths,
					params=PredictorParams()))
					
			self.load_predictor = load_predictor
			self.load_multi_predictor = load_multi_predictor
					
		else:
			from calamari_ocr.ocr import Predictor, MultiPredictor
			from calamari_ocr.ocr.voting.confidence_voter import ConfidenceVoter
		
			def load_predictor(path, **kwargs):
				return CalamariPredictor(Predictor(path, **kwargs))
				
			def load_multi_predictor(paths, **kwargs):
				return VotedCalamariPredictor(
					MultiPredictor(checkpoints=paths, **kwargs), ConfidenceVoter())

			self.load_predictor = load_predictor
			self.load_multi_predictor = load_multi_predictor



class TesseractPredictor:
	def __init__(self, kwargs):
		from tesserocr import PyTessBaseAPI, PSM
		self._api = PyTessBaseAPI(psm=PSM.SINGLE_LINE, **kwargs)

	def __call__(self, images, **kwargs):
		for image in images:
			self._api.SetImage(PIL.Image.fromarray(image))
			self._api.Recognize()
			yield self._api.GetUTF8Text()


from origami.batch.core.processor import Processor
from origami.batch.core.io import Artifact, Stage, Input, Output
from origami.batch.core.lines import LineExtractor
from origami.batch.core.utils import RegionsFilter


class OCRProcessor(Processor):
	def __init__(self, options):
		super().__init__(options)
		self._options = options
		self._backend = self._options["backend"]
		self._ocr = self._options["ocr"]
		self._predictor = None

		if self._ocr == "FAKE":
			self._model_path = None
			self._models = []
			self._line_height = 48
			self._chunk_size = 1
		elif self._backend == "calamari":
			if not options["model"]:
				raise click.BadParameter(
					"Please specify a model path", param="model")
			self._model_path = Path(options["model"])

			models = list(self._model_path.glob("*.json"))
			if not options["legacy_model"]:
				models = [m for m in models if m.with_suffix(".h5").exists()]
			if len(models) < 1:
				raise FileNotFoundError(
					"no Calamari models found at %s" % self._model_path)
			self._models = models

			self._line_height = None
			self._chunk_size = None
		elif self._backend == "tesseract":
			kwargs = {}
			if self._options["tess_data"]:
				kwargs["path"] = self._options["tess_data"]
			if self._options["tess_lang"]:
				kwargs["lang"] = self._options["tess_lang"]
			self._predictor = TesseractPredictor(kwargs)
			self._predict_kwargs = {}
			self._line_height = 48
			self._chunk_size = 1
		else:
			raise RuntimeError(f"unsupported ocr backend {self._backend}")

		self._ignored = RegionsFilter(options["ignore"])

		if self._ocr != "FULL":
			logging.getLogger().setLevel(logging.INFO)

	@staticmethod
	def options(f):
		options = [
			click.option(
				'-a', '--backend',
				required=False,
				type=click.Choice(['calamari', 'tesseract'], case_sensitive=False),
				default='calamari',
				help='ocr backend to use'),
			click.option(
				'-m', '--model',
				required=False,
				type=click.Path(exists=True),
				help='path that contains Calamari model(s)'),
			click.option(
				'--tess-lang',
				required=False,
				default='eng',
				type=str,
				help='language to use when using Tesseract'),
			click.option(
				'--tess-data',
				required=False,
				default='',
				type=str,
				help='tessdata path to use when using Tesseract'),
			click.option(
				'--legacy-model',
				is_flag=True,
				default=False,
				help='support Calamari legacy models (pre 1.0)'),
			click.option(
				'-b', '--batch-size',
				type=int,
				default=-1,
				required=False,
				help='batch size to use for Calamari'),
			click.option(
				'--ignore',
				type=str,
				default="regions/ILLUSTRATION"),
			click.option(
				'--ocr',
				type=click.Choice(['FULL', 'DRY', 'FAKE'], case_sensitive=False),
				default="FULL")
		]

		return functools.reduce(lambda x, opt: opt(x), options, f)

	@property
	def processor_name(self):
		return __loader__.name

	def _load_calamari_models(self):
		if self._backend != 'calamari':
			return
		
		if self._predictor is not None:
			return

		if self._ocr == "FAKE":
			return

		batch_size = self._options["batch_size"]
		if batch_size > 0:
			batch_size_kwargs = dict(batch_size=batch_size)
		else:
			batch_size_kwargs = dict()
		self._chunk_size = batch_size
		
		calamari = Calamari()

		if len(self._models) == 1:
			self._predictor = calamari.load_predictor(str(self._models[0]))
			self._predict_kwargs = batch_size_kwargs
			self._line_height = int(self._predictor.model_params.line_height)
		else:
			logging.info("using Calamari voting with %d models." % len(self._models))
			self._predictor = calamari.load_multi_predictor(
				[str(p) for p in self._models], **batch_size_kwargs)
			self._predict_kwargs = dict()
			self._line_height = int(self._predictor.predictors[0].model_params.line_height)

	def artifacts(self):
		return [
			("reliable", Input(
				Artifact.LINES, Artifact.TABLES,
				stage=Stage.RELIABLE)),
			("output", Output(Artifact.OCR)),
		]

	def process(self, page_path: Path, reliable, output):
		self._load_calamari_models()

		lines = reliable.lines.by_path

		extractor = LineExtractor(
			reliable.tables,
			self._line_height,
			self._options,
			min_confidence=reliable.lines.min_confidence)

		min_width = 6
		min_height = 6

		names = []
		empty_names = []
		images = []
		for stem, im in extractor(lines, ignored=self._ignored):
			if im.width >= min_width and im.height >= min_height:
				names.append("/".join(stem))
				images.append(np.array(im))
			else:
				empty_names.append("/".join(stem))

		if self._ocr == "DRY":
			logging.info("will ocr the following lines:\n%s" % "\n".join(sorted(names)))
			return

		chunk_size = self._chunk_size
		if chunk_size <= 0:
			chunk_size = len(images)

		texts = []

		if self._ocr == "FAKE":
			for name in names:
				texts.append("text for %s." % name)
		else:
			for i in range(0, len(images), chunk_size):
				for text in self._predictor(
						images[i:i + chunk_size], **self._predict_kwargs):
					texts.append(text)

		with output.ocr() as zf:
			for name, text in zip(names, texts):
				zf.writestr("%s.txt" % name, text)
			for name in empty_names:
				zf.writestr("%s.txt" % name, "")


@click.command()
@Processor.options
@LineExtractor.options
@OCRProcessor.options
def make_processor(**kwargs):
	return OCRProcessor(kwargs)


@click.command()
@click.argument(
	'data_path',
	type=click.Path(exists=True),
	required=True)
@Processor.options
@LineExtractor.options
@OCRProcessor.options
def run_ocr(data_path, **kwargs):
	""" Perform OCR on all recognized lines in DATA_PATH. """
	processor = OCRProcessor(kwargs)
	processor.traverse(data_path)


if __name__ == "__main__":
	run_ocr()


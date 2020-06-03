import ast
import re
import os
import click
import unicodedata

from pathlib import Path
from tqdm import tqdm


def _parse_str(s):
	if isinstance(s, str):
		return s
	elif s[0] == "chr":
		return chr(s[1])
	else:
		raise ValueError(s)


def _compile_rule(rule, schema):
	if rule[0] == "str":
		return lambda s: s.replace(_parse_str(rule[1]), _parse_str(rule[2]))
	elif rule[0] == "re":
		pattern = re.compile(rule[1])
		return lambda s: pattern.sub(rule[2], s)
	elif rule[0] == "tfm":
		return lambda s: schema.get_transformer(rule[1])(s)
	elif rule[0] == "unicode":
		return lambda s: unicodedata.normalize(rule[1], s)
	else:
		raise ValueError("illegal rule %s" % rule)


class Transformer:
	def __init__(self, schema, rules):
		self._rules = []
		for rule in rules:
			self._rules.append(_compile_rule(rule, schema))

	def __call__(self, text):
		for rule in self._rules:
			text = rule(text)
		return text.strip()


class Channel:
	def __init__(self, name, alphabet, transform, tests):
		self._name = name
		self._alphabet = set([c for c in alphabet]) if alphabet else None
		self._transform = transform
		self._tests = tests

	@property
	def name(self):
		return self._name

	@property
	def tests(self):
		return self._tests

	def transform(self, text):
		output_text = self._transform(text)

		if self._alphabet:
			alphabet = self._alphabet
			for i, c in enumerate(output_text):
				if c not in alphabet:
					err_text = "".join([
						output_text[:i],
						click.style(">", fg="red"),
						c,
						click.style("(0x%x)" % ord(c), fg="green"),
						click.style("<", fg="red"),
						output_text[i + 1:]])
					click.echo(err_text)

					raise click.ClickException(
						"encountered illegal characters in transcription.")

		return output_text

	def run_test(self, test_name, test_rules):
		n_fail = 0
		for input_text, expected_text in test_rules:
			output_text = self.transform(input_text)
			if output_text != expected_text:
				print("FAIL:")
				print("    computed: %s" % output_text)
				print("    expected: %s" % expected_text)
				n_fail += 1
		if n_fail == 0:
			status = "OK"
		else:
			status = "FAIL"
		print("%s TEST channel '%s' (test %s)" % (status, self.name, test_name))
		return n_fail == 0


class Schema:
	def __init__(self, path=None):
		if path is None:
			script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
			path = script_dir / "schema.txt"  # default schema

		with open(path, "r") as f:
			data = ast.literal_eval(f.read())

		self._tests = data["tests"]

		self._transformers = dict()
		for name, rules in data["transforms"].items():
			self._transformers[name] = Transformer(self, rules)

		self._channels = []
		for k, v in data["channels"].items():
			self._channels.append(Channel(
				name=k,
				alphabet=v.get("alphabet", None),
				transform=self.get_transformer(v.get("transform", None)),
				tests=v.get("tests", [])))

		self._run_tests()

	def _run_tests(self):
		all_ok = True
		for channel in self._channels:
			for test_name in channel.tests:
				all_ok = channel.run_test(
					test_name, self._tests[test_name]) and all_ok
		if not all_ok:
			raise RuntimeError("schema transformer tests failed.")

	def get_transformer(self, name):
		if name is None:
			return lambda text: text
		else:
			return self._transformers[name]

	@property
	def channels(self):
		return self._channels


@click.command()
@click.argument(
	'gt-path',
	type=click.Path(exists=True))
@click.option(
	'-s', '--schema-path',
	type=click.Path(exists=True),
	help='path to normalization schema',
	required=True)
@click.option(
	'-o', '--output-path',
	type=click.Path(exists=False),
	help='where to store normalized gt',
	required=True)
@click.option(
	'-e', '--extension',
	type=str,
	default=".gt.txt",
	help='which text files to process',
	required=True)
def cli(gt_path, schema_path, output_path, extension):
	output_path = Path(output_path).resolve()
	gt_path = Path(gt_path).resolve()
	assert gt_path != output_path

	schema = Schema(Path(schema_path))

	if len(schema.channels) != 1:
		raise RuntimeError("illegal number of channels in schema")
	channel = schema.channels[0]

	paths = [p for p in gt_path.iterdir() if p.name.endswith(extension)]
	normalized = dict()
	for p in tqdm(paths, desc="reading"):
		with open(p, "r") as f:
			text = f.read()

		try:
			normalized[p.name] = channel.transform(text)
		except:
			click.echo("Error in line %s." % p.name)
			raise

	output_path.mkdir()

	for line_name, annotation in tqdm(normalized.items(), desc="writing"):
		with open(output_path / line_name, "w") as f:
			f.write(annotation)


if __name__ == "__main__":
	cli()


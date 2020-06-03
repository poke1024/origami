import click
import json
import re

from pathlib import Path
from tqdm import tqdm
import numpy as np

import shutil
import skimage.filters
import skimage.morphology
import random
import PIL.Image


def _discretize(values, n=3):
	thresholds = np.quantile(values, [x / n for x in range(1, n)])

	def to_bin(x):
		for i, t in enumerate(thresholds):
			if x < t:
				return i
		return len(thresholds)

	return [to_bin(x) for x in values]


# from: https://stackoverflow.com/questions/1066758/
# find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def _rle(inarray):
	""" run length encoding. Partial credit to R rle function.
		Multi datatype arrays catered for including non Numpy
		returns: tuple (runlengths, startpositions, values) """
	ia = np.asarray(inarray)                  # force numpy
	n = len(ia)
	if n == 0:
		return None, None, None
	else:
		y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
		i = np.append(np.where(y), n - 1)   # must include last element posi
		z = np.diff(np.append(-1, i))       # run lengths
		p = np.cumsum(np.append(0, z))[:-1] # positions
		return z, p, ia[i]


class LineClustering:
	def __init__(self):
		self._forms = []

	def add(self, im):
		pixels = np.array(im.convert("L"))
		h, w = pixels.shape

		thresh_sauvola = skimage.filters.threshold_sauvola(
			pixels, window_size=h // 2 + 1)
		binarized = pixels > thresh_sauvola

		black = False
		white = True

		run_lengths = {black: [], white: []}
		for row in binarized:
			z, _, a = _rle(row)
			for length, f in zip(z, a):
				run_lengths[f].append(length)

		self._forms.append(np.mean(run_lengths[black]))
		#v = np.quantile(run_lengths[black], np.linspace(0, 1, 3))
		#v /= np.linalg.norm(v)
		#self._forms.append([np.mean(run_lengths[black])])

	def labels(self, n=3):
		#clust = sklearn.cluster.SpectralClustering(n_clusters=3)
		#clust = sklearn.cluster.AgglomerativeClustering(
		#	n_clusters=None, distance_threshold=0.5, linkage="complete")
		#clust = sklearn.cluster.MeanShift()
		#clust = sklearn.cluster.OPTICS(min_samples=0.1)
		#clust = sklearn.cluster.DBSCAN(eps=2.0)
		#clust.fit(np.array(self._forms))
		#return clust.labels_

		return _discretize(self._forms, n)

	def save_as_groups(self, image_paths, output_path):
		output_path = Path(output_path)
		output_path.mkdir()
		for i, x in tqdm(enumerate(self.labels)):
			dst_path = output_path / str(x)
			dst_path.mkdir(exist_ok=True)
			src_path = Path(image_paths[i])
			shutil.copy(src_path, dst_path / src_path.name)


def optimal_split(texts, labels=None, train_ratio=0.8, preset=None, optimizer="mip"):
	alphabet = dict()

	for text in texts:
		for letter in text:
			if letter not in alphabet:
				alphabet[letter] = len(alphabet)

	if labels:
		unique_labels = set(labels)
		print("optimal_split uses %d additional labels." % len(unique_labels))
		for label in unique_labels:
			alphabet[("label", label)] = len(alphabet)

	counts = np.zeros((len(texts), len(alphabet)), dtype=np.uint8)

	for i, text in tqdm(list(enumerate(texts)), desc="counting"):
		for letter in text:
			j = alphabet[letter]
			counts[i, j] += 1

		if labels:
			j = alphabet[("label", labels[i])]
			counts[i, j] += 1

	total = np.sum(counts, axis=0, dtype=np.uint32)

	cons = np.zeros((len(alphabet), ), dtype=np.uint32)
	for j, x in enumerate(total):
		cons[j] = max(1, min(x - 1, int(x * train_ratio)))

	print("building model.")

	if optimizer == "scip":
		from pyscipopt import Model, quicksum
		model = Model()
		xs = [model.addVar(vtype="BINARY") for _ in range(len(texts))]

		if preset:
			for i in preset.get(True, []):
				model.addCons(xs[i] == 1)
			for i in preset.get(False, []):
				model.addCons(xs[i] == 0)

		for j in range(len(alphabet)):
			freq = quicksum([x * counts[i, j] for i, x in enumerate(xs)])
			model.addCons(freq >= cons[j])

		model.setObjective(quicksum(xs))
		model.setMinimize()

		model.optimize()
		sol = model.getBestSol()
		#print(sol)

		allocation = np.zeros((len(xs)), dtype=np.bool)
		for i, x in enumerate(xs):
			allocation[i] = sol[x] > 0.5
	elif optimizer == "mip":
		from mip import Model, MINIMIZE, CBC, BINARY, xsum
		model = Model(sense=MINIMIZE, solver_name=CBC)
		xs = [model.add_var(var_type=BINARY) for _ in range(len(texts))]

		if preset:
			for i in preset.get(True, []):
				model += xs[i] == 1
			for i in preset.get(False, []):
				model += xs[i] == 0

		for j in range(len(alphabet)):
			freq = xsum(x * counts[i, j] for i, x in enumerate(xs))
			model += freq >= cons[j]

		model += xsum(xs) >= int(len(xs) * train_ratio)

		model.objective = xsum(xs)  # minimize
		status = model.optimize(max_seconds=2)
		print(status)

		allocation = np.zeros((len(xs)), dtype=np.bool)
		for i, x in enumerate(xs):
			allocation[i] = x.x > 0.5
	else:
		raise ValueError("unsupported optimizer %s" % optimizer)

	return allocation


def write_sets_json(out_path, allocation):
	with open(out_path, "w") as f:
		sets = dict(t=[], v=[], tv=[])

		for gt_name, is_train in allocation.items():
			sets["t" if is_train else "v"].append(gt_name)

		f.write(json.dumps(sets))


def gt_name(path):
	return re.sub(r"\.gt\.txt", "", path.name)


def find_image_path(images_path, gt_text_path):
	stem = gt_text_path.name.split(".")[0]

	for suffix in (".nrm.png", ".bin.png", ".png"):
		image_path = images_path / (stem + suffix)
		if image_path.exists():
			return image_path

	raise RuntimeError("image not found for %s" % gt_text_path)


@click.command()
@click.argument(
	'gt-path',
	type=click.Path(exists=True))
@click.option(
	'--optimize',
	type=click.Choice(['off', 'fast', 'best'], case_sensitive=False),
	default='off',
	help='optimize training set selection')
@click.option(
	'--val-size',
	type=float,
	default=0.2,
	help='relative size of validation set')
@click.option(
	'--image-path',
	type=click.Path(exists=True),
	help='path to line images')
def cli(gt_path, optimize, val_size, image_path, n_image_groups=5):
	paths = []
	for p in Path(gt_path).iterdir():
		if p.name.endswith(".gt.txt"):
			paths.append(p)

	random.seed(hash("here be dragons."))
	if optimize == "off":
		k = int(val_size * len(paths))
		random.shuffle(paths)

		allocation = dict()
		for p in paths[:k]:
			allocation[gt_name(p)] = False  # validation.
		for p in paths[k:]:
			allocation[gt_name(p)] = True  # training.

		write_sets_json(Path(gt_path) / "sets_random.json", allocation)
	else:
		texts = []
		for p in tqdm(paths, desc="texts"):
			with open(p, "r") as f:
				texts.append(f.read())

		labels = None

		if optimize == "best":
			if image_path is None:
				raise RuntimeError("image-path needs to be set.")
			image_path = Path(image_path)

			images = LineClustering()
			for p in tqdm(paths, desc="images"):
				im_p = find_image_path(image_path, p)
				images.add(PIL.Image.open(im_p))
			labels = images.labels(n_image_groups)

		allocation = optimal_split(
			texts, labels=labels, train_ratio=1 - val_size)

		n_train = sum(allocation)
		n_val = len(texts) - n_train

		print("train set size is %d." % n_train)
		print("val set size is %d." % n_val)
		print("val size ratio is %.1f." % (n_val / len(texts)))

		name_allocation = dict()
		for p, f in zip(paths, allocation):
			name_allocation[gt_name(p)] = f

		if optimize == "fast":
			opt_index = 1
		elif optimize == "best":
			opt_index = 2
		else:
			raise ValueError(optimize)

		write_sets_json(Path(gt_path) / f"sets_opt{opt_index}.json", name_allocation)


if __name__ == "__main__":
	cli()

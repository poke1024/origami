import numpy as np


class Label:
	def __init__(self, data, name, index):
		self._name = name
		self._color = data["rgbColor"]
		self._weight = data["weight"]
		self._index = index
		self._sep = data.get("separator")
		assert 0 <= index < 256
		assert len(self._color) == 3

	@property
	def name(self):
		return self._name

	def __hash__(self):
		return hash(self._name)

	def __eq__(self, other):
		return self.name == other.name

	@property
	def index(self):
		return self._index

	@property
	def color(self):
		return self._color

	@property
	def weight(self):
		return self._weight

	@property
	def is_separator(self):
		return self._sep is not None

	def is_separator_with_orientation(self, o):
		return self._sep and self._sep["orientation"].upper() == o.upper()

	@property
	def merge_distance(self):
		return self._sep.get("merge_distance", 500)

	@property
	def min_length(self):
		return self._sep.get("min_length", 0)


class LabelSet:
	def __init__(self, labels_json):
		self._labels_json = labels_json

		by_name = dict()
		by_index = dict()
		for i, name in enumerate(sorted(labels_json["annotations"].keys())):
			data = labels_json["annotations"][name]
			label = Label(data, name=name, index=i)
			by_name[name] = label
			by_index[i] = label

		self._by_name = by_name
		self._by_index = by_index

		groups = dict()
		for name, items in labels_json["groups"].items():
			group = []
			for item in items:
				group.append(by_name[item])
			groups[name] = group

		self._codes = dict(layout=groups)

		palette = np.zeros((3 * 256,), dtype=np.uint8)
		for name, label in by_name.items():
			i = label.index * 3
			palette[i:i + 3] = label.color
		self._palette = palette

		self._background = by_name["BACKGROUND"]

	def settings(self, key):
		return self._labels_json[key]

	@property
	def labels(self):
		return self._by_name.values()

	def label_from_name(self, name):
		return self._by_name[name]

	def label_from_index(self, index):
		return self._by_index[index]

	@property
	def n_labels(self):
		return len(self._by_name)

	@property
	def palette(self):
		return self._palette

	@property
	def background(self):
		return self._background

	@property
	def codes(self):
		return self._codes

	def separators(self, orientation):
		return [
			label for label in self._by_name.values()
			if label.is_separator_with_orientation(orientation)]

	@property
	def label_weights(self):
		n_labels = self.n_labels
		label_weights = np.empty(shape=(n_labels, ), dtype=np.float32)
		label_weights.fill(1)
		for label in self.labels:
			label_weights[label.index] = label.weight
		return label_weights

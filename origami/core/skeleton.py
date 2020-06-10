# see https://github.com/poke1024/fastskeleton

import skimage.measure
import scipy.ndimage
import numpy as np
import networkx as nx

from skimage.morphology import skeletonize
from numba import njit


def _build_neighborhood(kernel, n):
	a = np.empty((3, 3), dtype=np.bool)
	for i in range(3):
		for j in range(3):
			a[i, j] = 1 if (n & kernel[i, j] != 0) else 0
	return a


dx_dy = np.array(
	[(dx, dy)
	for dx in (-1, 0, 1)
	for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)])


@njit
def _traceback(labels, x, y):
	w, h = labels.shape[:2]
	l1, d1 = labels[x, y]
	pts = np.empty((d1 + 1, 2), dtype=np.int16)
	pts[d1] = (x, y)
	while d1 > 0:
		for dx, dy in dx_dy:
			xx = x + dx
			if xx < 0 or xx >= w:
				continue
			yy = y + dy
			if yy < 0 or yy >= h:
				continue

			l2, d2 = labels[xx, yy]
			if l2 == l1 and d2 < d1:
				for i in range(d2, d1):
					pts[i] = (x, y)
				d1 = d2
				x = xx
				y = yy
				break
	return pts


@njit
def _compute_graph_2(skel, labels, queues, links, find_paths):
	graph = dict()
	w, h = skel.shape

	while True:
		changed = False

		for qi, head in enumerate(queues):
			qx, qy = head
			next_ = (-1, -1)

			while qx >= 0 and qy >= 0:
				_, d = labels[qx, qy]

				for dx, dy in dx_dy:
					nx = qx + dx
					if nx < 0 or nx >= w:
						continue

					ny = qy + dy
					if ny < 0 or ny >= h:
						continue

					if not skel[nx, ny]:
						continue

					label, _ = labels[nx, ny]
					if label >= 0:
						u, v = qi, label
						if u != v:
							if u > v:
								u, v = v, u

							if (u, v) not in graph:
								if find_paths:
									t1 = _traceback(labels, qx, qy)
									t2 = _traceback(labels, nx, ny)
									pts = np.empty((len(t1) + len(t2), 2), dtype=np.int16)
									pts[:len(t1)] = t1
									pts[len(t1):] = t2[::-1]
								else:
									pts = np.empty((0, 2), dtype=np.int16)
								graph[(u, v)] = pts

						continue

					# claim label.
					labels[nx, ny] = (qi, d + 1)
					changed = True

					# enqueue.
					links[nx, ny] = next_
					next_ = (nx, ny)

				qx, qy = links[qx, qy]

			queues[qi] = next_

		if not changed:
			break

	return graph


def _compute_graph_data(skel, nodes, find_paths):
	w, h = skel.shape

	assert w < np.iinfo(np.int16).max
	assert h < np.iinfo(np.int16).max

	if w * h >= np.iinfo(np.int16).max:
		label_type = np.int32
	else:
		label_type = np.int16

	labels = np.empty((w, h, 2), dtype=label_type)
	labels.fill(-1)

	links = np.empty((w, h, 2), dtype=np.int16)
	links.fill(-1)

	queues = np.empty((len(nodes), 2), dtype=np.int16)
	for i, node in enumerate(nodes):
		queues[i] = nodes[i]
		x, y = node
		labels[x, y] = (i, 0)
		assert skel[x, y]

	return _compute_graph_2(skel, labels, queues, links, find_paths)


class FastSkeleton:
	def __init__(self):
		kernel = np.array([
			[0x01, 0x02, 0x04],
			[0x08, 0x00, 0x10],
			[0x20, 0x40, 0x80]
		])
		self._kernel = kernel

		nhood_comp = np.empty((2 ** 9,), dtype=np.int8)
		for i in range(2 ** 9):
			a = _build_neighborhood(kernel, i)
			_, num = skimage.measure.label(
				a, return_num=True, connectivity=1)
			nhood_comp[i] = num
		self._nhood_comp = nhood_comp

	def __call__(self, pixels, paths=True, time=False):
		if not type(pixels) is np.ndarray or pixels.dtype != np.bool:
			raise ValueError("pixels needs to be a boolean numpy array")

		skeleton = skeletonize(pixels)

		nhood = scipy.ndimage.convolve(
			skeleton.astype(np.uint8),
			self._kernel,
			mode='constant',
			cval=0)

		n_comp = self._nhood_comp[nhood]
		nodes = np.transpose(np.nonzero(np.logical_and(
			n_comp != 2, skeleton)))

		graph_data = _compute_graph_data(skeleton, nodes, paths)

		if time:
			edt = scipy.ndimage.morphology.distance_transform_edt(pixels)

		nodes = [tuple(pt) for pt in np.flip(nodes, axis=-1)]
		graph = nx.Graph()
		graph.add_nodes_from(nodes)

		if time:
			attr = dict(((x, y), dict(time=edt[y, x])) for x, y in nodes)
			nx.set_node_attributes(graph, attr)

		for (i, j), path in graph_data.items():
			kwargs = dict()
			if paths:
				pts = np.flip(path, axis=-1)
				kwargs["path"] = [tuple(pt) for pt in pts]
				d = [np.linalg.norm(p - q) for p, q in zip(pts, pts[1:])]
				kwargs["distance"] = np.sum(d)
			if time:
				kwargs["time"] = edt[tuple(np.transpose(path))]
			graph.add_edge(nodes[i], nodes[j], **kwargs)

		return graph

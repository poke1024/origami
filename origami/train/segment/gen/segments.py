import numpy as np
import scipy
import math
import cv2
import shapely.geometry
import enum


def mask_to_polyline_robust(mask, label, accuracy=5):
	pts = np.argwhere(mask).astype(np.float32)
	pts = np.flip(pts, -1)

	vx, vy, cx, cy = cv2.fitLine(
		pts, cv2.DIST_L2, 0, 0.01, 0.01)

	if label.is_separator_with_orientation("H"):
		if vx < 0:  # always point right
			vy = -vy
			vx = -vx
	else:
		if vy < 0:  # always point down
			vy = -vy
			vx = -vx

	x, y = np.transpose(pts, axes=(1, 0))

	qx = x - cx
	qy = y - cy

	sv = qx * vx + qy * vy

	ux = -vy
	uy = vx

	su = qx * ux + qy * uy

	tv = (np.min(sv), np.max(sv))
	tu = (np.min(su), np.max(su))

	num = max(math.ceil((tv[1] - tv[0]) / accuracy), 3)

	p = np.array([cx, cy]).flatten()
	v = np.array([vx, vy]).flatten()
	u = np.array([-vy, vx]).flatten()

	r = []
	widths = []

	t = np.linspace(tv[0], tv[1], num=num)
	for t0, t1 in zip(t, t[1:]):
		mask = np.logical_and(sv >= t0, sv <= t1)
		if np.sum(mask.astype(np.uint8)) > 0:
			part_sv = np.expand_dims(sv[mask], axis=0).T
			part_su = np.expand_dims(su[mask], axis=0).T
			pts = p + v * part_sv + u * part_su
			r.append(np.median(pts, axis=0))
			widths.append(np.median(abs(part_su)))

	if True and len(r) > 5:
		r = np.array(r)
		x = _running_mean(r[:, 0], 5)
		y = _running_mean(r[:, 1], 5)

		x = list(r[:2, 0]) + list(x) + list(r[-2:, 0])
		y = list(r[:2, 1]) + list(y) + list(r[-2:, 1])
		r = np.array([x, y]).T

		'''
		r = np.array(r)
		x = _running_mean(r[:, 0], 3)
		y = _running_mean(r[:, 1], 3)

		x = [r[0, 0]] + list(x) + [r[-1, 0]]
		y = [r[0, 1]] + list(y) + [r[-1, 1]]
		r = np.array([x, y]).T
		'''

	lines = shapely.geometry.LineString(r)
	lines = lines.simplify(0.5, preserve_topology=False)

	thickness = np.median(widths)
	err = thickness / (tv[1] - tv[0])

	return lines, max(1, thickness), err


def mask_to_polyline(mask, label):
	return mask_to_polyline_robust(mask, label)


def _running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)


def smoothened_at(pts, i):
	if i < 3 or len(pts) < 3 + i:
		return pts

	pts = np.array(pts.copy())

	x = _running_mean(pts[:, 0], 5)
	y = _running_mean(pts[:, 1], 5)

	k = i - 2 - 2
	if k < 0:
		return pts

	n = len(x[k:k + 5])
	n = min(n, len(pts[i - 2:i - 2 + n, 0]))

	if n < 1:
		return pts

	try:
		pts[i - 2:i - 2 + n, 0] = x[k:k + n]
		pts[i - 2:i - 2 + n, 1] = y[k:k + n]
	except ValueError:
		print(len(pts), len(x), i, len(x[k:k + 5]))
		raise

	return pts


class JoinResult(enum.Enum):
	OK = 0
	LABEL_FAIL = 1
	COLLAPSE_FAIL = 2
	PARALLEL_FAIL = 3
	DISTANCE_FAIL = 4
	MASK_FAIL = 5
	DIRECTION_FAIL = 6


class Segment:
	def __init__(self, segment=None):
		self._label = segment._label if segment else None
		self._mask = None
		self._path = None
		self._thickness = None
		self._err = None
		self._name = None
		self._xpath = None

	@property
	def path(self):
		return self._path

	@property
	def centre(self):
		return self._path.centroid.coords[0]

	@property
	def name(self):
		return self._name

	@property
	def thickness(self):
		return self._thickness

	@staticmethod
	def from_mask(label_set, labels, mask, name):
		segment = Segment()

		segment._label = label_set.label_from_index(np.argmax(np.bincount(labels[mask])))
		segment._mask = mask
		segment._path, segment._thickness, segment._err = mask_to_polyline(mask, segment._label)
		segment._name = name

		return segment

	@staticmethod
	def _are_vectors_parallel(a, b):
		la, lb = np.linalg.norm(a), np.linalg.norm(b)
		length = min(la, lb)

		if length < 3:
			return True  # no reliable direction here.
		elif length < 50:
			# SNP2436020X-19200601-0-0-0-0.02 top right segment patch.
			phi_threshold = 30
		else:
			phi_threshold = 15

		try:
			dot = np.dot(a / la, b / lb)
			phi = np.degrees(abs(math.acos(dot)))
		except ValueError:
			return False

		if phi > 90:
			phi = 180 - phi
		if phi > phi_threshold:
			return False

		return True

	@staticmethod
	def parallel_and_close(a, b, distance):
		if not Segment._are_vectors_parallel(a.v, b.v):
			return False
		elif a.path.distance(b.path) > distance:
			return False
		else:
			return True

	def transform(self, t):
		segment = Segment(self)

		if self._mask is not None:
			segment._mask = t.mask(self._mask)

		segment._path = t.geometry(self._path)

		# we assume that m is only a rotation.

		segment._thickness = self._thickness
		segment._err = self._err
		segment._name = self._name

		return segment

	@property
	def p(self):
		return np.array(self._path.coords[0])

	@property
	def v(self):
		v = np.array(self._path.coords[-1]) - np.array(self._path.coords[0])
		length = np.linalg.norm(v)
		if length == 0:
			return np.array([0, 0])
		else:
			return v / length

	@property
	def u(self):
		v = self.v
		return np.array([-v[1], v[0]])

	@property
	def angle(self):
		vx, vy = self.v
		return np.degrees(math.atan2(vy, vx))

	@property
	def error(self):
		return self._err

	@property
	def length(self):
		return self._path.length

	@property
	def mask(self):
		return self._mask

	@property
	def endpoints(self):
		a = self._path.coords[0]
		b = self._path.coords[-1]
		return np.array(a), np.array(b)

	def endpoint_i(self, p):
		for i, q in enumerate(self.endpoints):
			if all(p == q):
				return i
		return None

	@property
	def dominant_label(self):
		return self._label

	@property
	def dewarped_v(self):
		if self.dominant_label.is_separator_with_orientation("H"):
			v = [np.sign(self._v[0]), 0]
		else:
			v = [0, np.sign(self._v[1])]

		return np.array(v, dtype=np.float32).flatten()

	@property
	def dewarped(self):
		line = (self._p, self.dewarped_v)

		segment = Segment(self)

		segment._p, segment._v = self._p, self.dewarped_v

		segment._tv = self._tv
		segment._tu = self._tu

		segment._len = self._len
		segment._err = self._err

		segment._sv = None
		segment._su = None

		return segment

	def extend_by(self, amount):
		if amount == 1:
			return self
		else:
			spl, (t0, t1) = self.spline

			length = t1 - t0
			t0 -= length * amount / 2
			t1 += length * amount / 2

			t = np.linspace(t0, t1, math.ceil(len(self.path.coords) * (1 + amount)))
			s = scipy.interpolate.splev(t, spl)

			vv = self.v * np.array([t]).T
			uu = self.u * np.array([s]).T

			segment = Segment()

			segment._label = self._label
			segment._mask = self._mask  # not extended
			segment._path = shapely.geometry.LineString(self.p + vv + uu)
			segment._thickness = self._thickness
			segment._err = self._err
			segment._name = self._name

			return segment

	def _extrapolate(self, spl, shape, num=10):
		p = self.p
		v = self.v
		u = self.u

		dx = abs(v[0])
		dy = abs(v[1])

		if self.dominant_label.is_separator_with_orientation("H"):
			t0 = -p[0] / dx
			t1 = (shape[1] - p[0]) / dx
		else:
			t0 = -p[1] / dy
			t1 = (shape[0] - p[1]) / dy

		t = np.linspace(t0, t1, num)
		s = scipy.interpolate.splev(t, spl)

		vv = v * np.array([t]).T
		uu = u * np.array([s]).T

		return shapely.geometry.LineString(self.p + vv + uu)

	def _estimate_thickness(self):
		spl = self.spline
		self._mask

	@property
	def spline(self):
		#spl = scipy.interpolate.UnivariateSpline(pts[0], pts[1], k=2)
		#spl.set_smoothing_factor(0.5)

		pts = list(self._path.simplify(10, preserve_topology=False).coords)
		pts = np.array(pts)

		p = self.p
		v = self.v
		u = self.u

		lpts = pts - p

		x = lpts.dot(np.array([v]).T).flatten()
		y = lpts.dot(np.array([u]).T).flatten()

		i = np.argsort(x)
		x = x[i]
		y = y[i]

		m = lpts.shape[0]

		if m < 10:
			k = 1
		else:
			k = 2

		try:
			spl = scipy.interpolate.splrep(
				x, y, k=k, s=m + math.sqrt(2 * m))
		except ValueError:
			print("error in splrep", x, y, m)
			return None

		return spl, (np.min(x), np.max(x))

	def compute_xpath(self, shape):
		spl, _ = self.spline
		self._xpath = self._extrapolate(spl, shape)

	@property
	def xpath(self):
		return self._xpath


class SegmentJoiner:
	def __init__(self, spec, label_set):
		self._spec = spec
		self._label_set = label_set

	def join_mask_counts(self, dominant_label, labels, a, b, thickness, debug=None):
		join_mask = np.zeros(labels.shape, dtype=np.uint8)
		cv2.line(
			join_mask,
			tuple(a.astype(np.int32)),
			tuple(b.astype(np.int32)),
			1,
			int(max(1, math.ceil(thickness))))
		join_mask = join_mask > 0

		if debug:
			debug(dominant_label, join_mask)

		n_labels = self._label_set.n_labels
		counts = np.bincount(labels[join_mask], minlength=n_labels)
		counts = counts.astype(np.float32) / float(np.sum(counts))
		return counts, join_mask

	def check_join_mask(self, dominant_label, labels, a, b, thickness, ignore=[], debug=None):
		counts, join_mask = self.join_mask_counts(
			dominant_label, labels, a, b, thickness, debug)
		# print(counts)

		for label in ignore:
			counts[label.index] = 0

		for item in self._spec:
			if dominant_label.name == item.get("separator"):
				l1 = self._label_set.label_from_name(item["separator"])
				l2 = self._label_set.label_from_name(item["may_cross"])
				if counts[l1.index] > counts[l2.index]:
					counts[l2.index] = 0
			elif "separator" not in item:
				l2 = self._label_set.label_from_name(item["may_cross"])
				counts[l2.index] = 0

		counts[self._label_set.background.index] = 0
		counts[dominant_label.index] = 0

		# print(counts)
		if np.sum(counts) > 0:
			return False, join_mask
		else:
			return True, join_mask

	def join(self, labels, a, b, indices=None, debug=None):
		if a.dominant_label != b.dominant_label:
			return JoinResult.LABEL_FAIL, None

		a_pts = a.endpoints
		b_pts = b.endpoints

		if all(a_pts[0] == b_pts[0]) or all(a_pts[1] == b_pts[1]):
			return JoinResult.COLLAPSE_FAIL, None

		if indices is None:
			distances = [(np.linalg.norm(p1 - p2), i1, i2)
						 for i1, p1 in enumerate(a_pts)
						 for i2, p2 in enumerate(b_pts)]

			indices = sorted(distances, key=lambda d: d[0])[0][1:]

		# note: indices might not always be the best distance here.
		# we might have checked the shortest distance earlier and
		# already rejected it.

		a_i, b_i = indices

		vv = b_pts[b_i] - a_pts[a_i]
		#vv /= np.linalg.norm(vv)

		if a.dominant_label.is_separator_with_orientation("H"):
			dominant_dir = np.float32([1, 0])
		else:
			dominant_dir = np.float32([0, 1])
		tangent = vv / np.linalg.norm(vv)
		# endpoints are ordered in dominant_dir.
		if a_i == 1:
			# b must lie in direction of dominant_dir.
			if tangent.dot(dominant_dir) < 0:
				return JoinResult.DIRECTION_FAIL, False
		else:
			if tangent.dot(dominant_dir) > 0:
				return JoinResult.DIRECTION_FAIL, False

		v0 = a.v if a.length > b.length else b.v
		#v0 /= np.linalg.norm(v0)

		#if a.dominant_label == Label.H:
		#    print("!", vv, v0)

		#if (vv / np.linalg.norm(vv)).dot(v0 / np.linalg.norm(v0)) < 0:
		#    return JoinResult.PARALLEL_FAIL, None

		if not Segment._are_vectors_parallel(v0, vv):
			return JoinResult.PARALLEL_FAIL, None

		v0 /= np.linalg.norm(v0)
		orth_dist = abs(np.dot(a_pts[0] - b_pts[0], np.array([-v0[1], v0[0]])))
		# SNP2436020X-19200601-0-0-0-0.02 top right segment patch.
		if orth_dist > 25:
			return JoinResult.DISTANCE_FAIL, None

		thickness = (a.thickness * a.length + b.thickness * b.length) / (a.length + b.length)

		ok, join_mask = self.check_join_mask(
			a.dominant_label, labels,
			a_pts[a_i], b_pts[b_i],
			thickness, debug=debug)
		if not ok:
			return JoinResult.MASK_FAIL, None

		if False:
			# super BBZ-specific stuff here.

			if a.dominant_label == Label.TABCOL:
				# prevent table columns to extend through text blocks,
				# see e.g. spilling over "Erze." in SNP2436020X-19200601-1-0-0-0.03
				# or "Noten." in 2436020X_1925-02-27_70_98_006

				test_tt = 10

				for test_t in (100,):  # 500, 400, 300, 200, 100, 50):
					counts, _ = self.join_mask_counts(
						a.dominant_label, labels, a_pts[a_i], b_pts[b_i], test_t)
					if counts[int(Label.V)] == 0.:
						test_tt = test_t
						break

				ok, _ = self.check_join_mask(
					a.dominant_label, labels,
					a_pts[a_i], b_pts[b_i],
					test_tt, ignore=[Label.TABTXT, Label.H], debug=debug)
				if not ok:
					return JoinResult.MASK_FAIL, None

		a_coords = list(a.path.coords)
		b_coords = list(b.path.coords)
		if a_i == 1 and b_i == 0:
			j_coords = a_coords + b_coords
			j_coords = smoothened_at(j_coords, len(a_coords))
		elif b_i == 1 and a_i == 0:
			j_coords = b_coords + a_coords
			j_coords = smoothened_at(j_coords, len(b_coords))
		else:  # should not happen
			print("WARN", "illegal order", a_i, b_i)
			if a_i == 0:
				a_coords = list(reversed(a_coords))
			if b_i == 1:
				b_coords = list(reversed(b_coords))
			j_coords = a_coords + b_coords
			j_coords = smoothened_at(j_coords, len(a_coords))

		try:
			assert any(j_coords[0] != a_pts[a_i])
			assert any(j_coords[-1] != a_pts[a_i])
			assert any(j_coords[0] != b_pts[b_i])
			assert any(j_coords[-1] != b_pts[b_i])
		except:
			print("ERR", "j", j_coords[0], j_coords[-1], "a_i", a_i, "a_pts", a_pts)
			print("a_coords", a_coords)
			print("b_coords", b_coords)
			raise

		joined = Segment(a)
		joined._mask = np.logical_or.reduce([a._mask, b._mask, join_mask])
		joined._path = shapely.geometry.LineString(j_coords)
		joined._thickness = thickness
		joined._err = min(a._err, b._err)
		joined._name = "%s-%s" % (a._name, b._name)

		return JoinResult.OK, joined

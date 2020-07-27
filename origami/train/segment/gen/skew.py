import numpy as np
import skimage
import math


def estimate_angle(coords, orthogonal=False):
	coords = np.array(coords)

	if len(coords) < 3:
		return False

	x0 = coords[0, 0]
	x1 = coords[-1, 0]
	y0 = coords[0, 1]
	y1 = coords[-1, 1]

	try:
		if abs(x1 - x0) > abs(y1 - y0):
			model, _ = skimage.measure.ransac(
				coords,
				skimage.measure.LineModelND,
				min_samples=2,
				residual_threshold=1,
				max_trials=1000)

			y0, y1 = model.predict_y([x0, x1])
			vy = y1 - y0
			vx = x1 - x0
			phi = math.pi / 2 - math.atan2(vy, vx)
		else:
			model, _ = skimage.measure.ransac(
				np.flip(coords, -1),
				skimage.measure.LineModelND,
				min_samples=2,
				residual_threshold=1,
				max_trials=1000)

			x0, x1 = model.predict_y([y0, y1])
			vy = y1 - y0
			vx = x1 - x0
			phi = math.pi / 2 + math.atan2(vy, vx)

	except ValueError:
		return False

	if orthogonal:
		phi -= math.pi / 2

	phi = math.asin(math.sin(phi))  # limit to -pi/2, pi/2
	phi = np.degrees(phi)

	return phi


def estimate_skew(segments, max_skew=15):
	total_length = 0
	sum_of_angles = 0

	# newspaper pages are usually higher than wide. prefer V separators
	# for skew estimation here.
	if any(s.dominant_label.is_separator_with_orientation("V") for s in segments):
		selected_orientation = "V"
	else:
		selected_orientation = "H"

	for s in segments:
		if s.dominant_label.is_separator_with_orientation(selected_orientation):

			phi = estimate_angle(
				s.path.coords,
				s.dominant_label.is_separator_with_orientation("H"))

			if phi is False:
				continue

			if abs(phi) > max_skew:
				# there are such things as tables rotated by 90 degrees. so
				# be careful to not use these for skew estimation here.
				continue

			# print("!", s.dominant_label, phi, s.length, vx, vy)

			length = s.length

			total_length += length
			sum_of_angles += phi * length

	return sum_of_angles / total_length

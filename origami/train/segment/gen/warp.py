'''
this code is a modified version of https://github.com/mzucker/page_dewarp
by Matt Zucker
'''

import numpy as np
import cv2
import random
import hashlib

from .transform import Remap


def warp_images(ground_truth, label_set, name):
	def norm2pix(shape, pts, as_integer):
		height, width = shape[:2]
		scl = max(height, width) * 0.5
		offset = np.array([0.5 * width, 0.5 * height],
			dtype=pts.dtype).reshape((-1, 1, 2))
		rval = pts * scl + offset
		if as_integer:
			return (rval + 0.5).astype(int)
		else:
			return rval

	def pix2norm(shape, pts):
		height, width = shape[:2]
		scl = 2.0 / (max(height, width))
		offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
		return (pts - offset) * scl

	RVEC_IDX = slice(0, 3)  # index of rvec in params vector
	TVEC_IDX = slice(3, 6)  # index of tvec in params vector
	CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

	FOCAL_LENGTH = 1.2  # normalized focal length of camera

	# default intrinsic parameter matrix
	K = np.array([
		[FOCAL_LENGTH, 0, 0],
		[0, FOCAL_LENGTH, 0],
		[0, 0, 1]], dtype=np.float32)

	def project_xy(xy_coords, pvec):

		# get cubic polynomial coefficients given
		#
		#  f(0) = 0, f'(0) = alpha
		#  f(1) = 0, f'(1) = beta

		alpha, beta = tuple(pvec[CUBIC_IDX])

		poly = np.array([
			alpha + beta,
			-2 * alpha - beta,
			alpha,
			0])

		xy_coords = xy_coords.reshape((-1, 2))
		z_coords = np.polyval(poly, xy_coords[:, 0])

		objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

		image_points, _ = cv2.projectPoints(objpoints,
											pvec[RVEC_IDX],
											pvec[TVEC_IDX],
											K, np.zeros(5))

		return image_points

	def get_corners(x_dir, im_shape):

		if x_dir[0] < 0:
			x_dir = -x_dir

		y_dir = np.array([-x_dir[1], x_dir[0]])

		h, w = im_shape[:2]
		pagecoords = np.array([[w, h], [0, h], [0, 0], [w, 0]])
		pagecoords = pix2norm(im_shape, pagecoords.reshape((-1, 1, 2)))
		pagecoords = pagecoords.reshape((-1, 2))

		px_coords = np.dot(pagecoords, x_dir)
		py_coords = np.dot(pagecoords, y_dir)

		px0 = px_coords.min()
		px1 = px_coords.max()

		py0 = py_coords.min()
		py1 = py_coords.max()

		p00 = px0 * x_dir + py0 * y_dir
		p10 = px1 * x_dir + py0 * y_dir
		p11 = px1 * x_dir + py1 * y_dir
		p01 = px0 * x_dir + py1 * y_dir

		return np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

	def get_default_params(corners):

		# page width and height
		page_width = np.linalg.norm(corners[1] - corners[0])
		page_height = np.linalg.norm(corners[-1] - corners[0])
		rough_dims = (page_width, page_height)

		# our initial guess for the cubic has no slope
		cubic_slopes = [0.0, 0.0]

		# object points of flat page in 3D coordinates
		corners_object3d = np.array([
			[0, 0, 0],
			[page_width, 0, 0],
			[page_width, page_height, 0],
			[0, page_height, 0]])

		# estimate rotation and translation from four 2D-to-3D point
		# correspondences
		_, rvec, tvec = cv2.solvePnP(
			corners_object3d,
		 	corners, K, np.zeros(5))

		params = np.hstack((np.array(rvec).flatten(),
							np.array(tvec).flatten(),
							np.array(cubic_slopes).flatten()))

		return rough_dims, params

	class Warper:
		def __init__(self, shape):
			rough_dims, self._params = get_default_params(
				get_corners(np.array([1, 0]), shape))
			page_dims = rough_dims

			height = 0.5 * page_dims[1] * shape[0]
			height = int(height)
			width = int(height * page_dims[0] / page_dims[1])

			# HACK
			width = shape[1]
			height = shape[0]

			page_x_range = np.linspace(0, page_dims[0], width)
			page_y_range = np.linspace(0, page_dims[1], height)

			self._page_x_coords, self._page_y_coords = np.meshgrid(
				page_x_range, page_y_range)

			self._page_xy_coords = np.hstack((
				self._page_x_coords.flatten().reshape((-1, 1)),
				self._page_y_coords.flatten().reshape((-1, 1))))

			self._page_xy_coords = self._page_xy_coords.astype(np.float32)

			self._width = width
			self._height = height

		def __call__(self, kind, im, alpha=0.0, beta=0.0):
			params = self._params.copy()
			params[CUBIC_IDX] = (alpha, beta)

			image_points = project_xy(self._page_xy_coords, params)

			image_points = norm2pix(im.shape, image_points, False)

			image_x_coords = image_points[:, 0, 0].reshape(self._page_x_coords.shape)
			image_y_coords = image_points[:, 0, 1].reshape(self._page_y_coords.shape)

			image_x_coords = cv2.resize(
				image_x_coords, (self._width, self._height),
				interpolation=cv2.INTER_CUBIC)

			image_y_coords = cv2.resize(
				image_y_coords, (self._width, self._height),
				interpolation=cv2.INTER_CUBIC)

			remap = Remap(image_x_coords, image_y_coords)

			if kind == "labels":
				return remap.labels(im, label_set.label_weights, border=0)
			elif kind == "image":
				return remap.grayscale(im, border=(255, 255, 255)).astype(im.dtype)
			else:
				raise RuntimeError("unsupported data type %s" % kind)

	random.seed(hashlib.sha256(name.encode("utf8")).digest())
	alpha = random.uniform(-0.2, 0.2)
	beta = random.uniform(-0.2, 0.2)

	warper = Warper(ground_truth.shape)
	return ground_truth.transform(
		lambda kind, image: warper(kind, image, alpha, beta))

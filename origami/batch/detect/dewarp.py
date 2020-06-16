import numpy as np

from sklearn.decomposition import PCA

def _estimate_warp(self, seps):
	if not seps:
		return 0

	pca = PCA(n_components=2)
	err = []

	for sep in seps:
		coords = np.array(list(sep.coords))
		pca.fit(coords)

		err.append(min(pca.explained_variance_))

	return np.max(err)


	@property
	def warp(self):
		warps = []
		for seps in self._separators.values():
			warps.append(self._estimate_warp(seps))
		return max(warps)

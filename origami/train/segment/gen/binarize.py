import PIL.Image
import numpy as np
import skimage


def build_binarized(img_path, window_size=15):
	bin_path = img_path.parent.parent / "bin"
	if not bin_path.exists():
		bin_path.mkdir()
	assert bin_path.is_dir()

	out_bin_path = bin_path / (img_path.stem + ".png")

	if not out_bin_path.is_file():
		im = PIL.Image.open(img_path)

		pixels = np.array(im.convert("L"))

		thresh_sauvola = skimage.filters.threshold_sauvola(
			pixels, window_size=window_size)
		binarized = PIL.Image.fromarray(pixels > thresh_sauvola)

		binarized = binarized.convert('1')

		binarized.save(out_bin_path, "png")


def gen_binarized(path):
	for p in path.iterdir():
		if p.is_dir():
			gen_binarized(p)
		elif p.parent.name == "img" and not p.stem.startswith("."):
			build_binarized(p)


from origami.batch.detect.layout import *

_fringe = 0.001


def y_aligned(contours, text, table):
	_, miny1, _, maxy1 = contours[text].bounds
	_, miny2, _, maxy2 = contours[table].bounds
	return alignment(miny1, maxy1, miny2, maxy2, mode="a") > 0.9


_region_code = {
	("regions", "TEXT"): "txt",
	("regions", "TABULAR"): "tab"
}


def split_text_table(text, table):
	_, tab_miny, _, tab_maxy = table.bounds
	union = text.union(table)
	minx, miny, maxx, maxy = union.bounds

	table_dom = shapely.geometry.box(
		minx - 1, tab_miny, maxx + 1, tab_maxy)
	new_table_shape = union.intersection(table_dom)
	new_text_shape = union.difference(table_dom)
	return new_text_shape, new_table_shape


def dominance_strategy(contours, a, b):
	code = tuple([_region_code[x[:2]] for x in (a, b)])
	if code == ("txt", "tab"):
		if y_aligned(contours, a, b):
			return "merge", b
		else:
			r = split_text_table(contours[a], contours[b])
			return "custom", r
	elif code == ("tab", "txt"):
		if y_aligned(contours, b, a):
			return "merge", a
		else:
			r = split_text_table(contours[b], contours[a])
			return "custom", tuple(reversed(r))
	if contours[a].area < contours[b].area:
		return "split", b, a
	else:
		return "split", a, b


def make_transformer():
	seq_merger = SequentialMerger(
		filters="regions/TABULAR",
		cohesion=(0.5, 0.8),
		max_distance=0.01,
		max_error=0.05,
		fringe=_fringe,
		obstacles=["separators/V"])

	return Transformer([
		SetUnionOperator("convex"),
		Dilation("none"),
		AdjacencyMerger(
			"regions/TEXT",
			IsOnSameLine(
				max_line_count=3,
				fringe=_fringe)),
		OverlapMerger(0.1),
		Shrinker(),
		seq_merger,
		AdjacencyMerger(
			"regions/TABULAR", IsBelow()),
		seq_merger,
		OverlapMerger(0),
		Dilation("rect"),
		SetUnionOperator("none"),
		DominanceOperator(
			filters="regions/TEXT, regions/TABULAR",
			fringe=0,
			strategy=dominance_strategy),
		FixSpillOverH("regions/TEXT"),
		FixSpillOverV("regions/TEXT"),
		AreaFilter(0.0025)
	])

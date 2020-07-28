from origami.batch.detect.layout import *

_fringe = 0.001


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
            strategy="take_from_large"),
        FixSpillOverH("regions/TEXT"),
        FixSpillOverV("regions/TEXT"),
        AreaFilter(0.005)
    ])

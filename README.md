Divisive (but not hierarchical) clustering.

Slow, but clustering exactly how I wanted it. Iteratively split cluster with highest SSE. Splits are used to find new exemplars, which are thrown into k-medians with existing exemplars.

Advantages:
* finds big clusters
* deterministic
* robust to outliers
* fast for what it is, can easily split a few thousand members into a small number of clusters
* decisions made early on do not effect the result as much as they would in true hierarchical clustering
* has strategies to avoid ties or arbitrary (even if deterministic) decisions with small member sets

Disadvantages:
* child clusters will not necessarily contain (or only contain) the members of the parent, so this is not hierarchical (unlike agglomerative clustering where you can build a tree and then transverse it cheaply)
* as a result of not being hierarchical, it is not straightforward to undo changes if you end up with, for instance, two near-identical exemplars. This could be implemented, but isn't implemented here.
* slows down as the number of clusters grows, not the best way to de-cluster all the way back to constituent members
* can only use Euclidean distance; CIELab and other color-distance metrics (which aren't all they're cracked up to be anyway) will not work (that's a trade-off for some of the optimizations)

This clustering is designed for questions like "what are the two dominant colors in this image (respecting transparency)?"

Once you split a cluster, it's acceptable to just throw it away. And you will probably only need the exemplars of the clusters you do keep, though you could re-cluster each cluster's members for a semi-hierarchical clustering scheme.

## Three large steps in the background

### Average colors by n-bit representation

`pool_colors`: reduce 8-bit image colors (potentially 16_777_216 colors) to a maximum of 262_144 by averaging. The ouput of `pool_colors` will also contain a weight axis for each color, representing the combined opacity of all pixels of that color.

### Median cut along longest axis

`cut_colors`: reduce colors to around 512 by recursively splitting along longest axis (longest actual axis. Not constrained to x, y, or, z axes).

### k-medians clustering

`KMediansClusters`: split and merge clusters. Again this is *not* hierarchical, so the sequence

* start
* split
* merge

is not guaranteed to bring you back to start. The merge methods are "public", but their principal use is to break ties in order to maintain deterministic results. For example:

* start with one cluster with 100 members
* split this cluster recursively into five clusters (30, 30, 20, 10, 10)
* ask for the largest cluster, and there's a tie
* KMediansClusters will recursively merge the closest two clusters until the tie is broken

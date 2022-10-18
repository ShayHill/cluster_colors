Divisive (but not hierarchal) clustering.

Slow, but clustering exactly how I wanted it. Iteratively split cluster with highest SSE. Splits are used to find new examplars, which are thrown into kmedians with existing exemplars.

Advantages:
* finds big clusters
* deterministic
* robust to outliers
* fast for what it is, can easily split a few thousand members into a small number of clusters
* decisions made early on to not effect the result as much as they would in true hierarchal clustering
* has strategies to avoid ties or arbitrary (if deterministic) decisions with small member sets

Disadvantages:
* child clusters will not necessarily contain (or only contain) the members of the parent, so this is not hierarchal (unlike agglomerative clustering where you can build a tree and then transverse it cheaply)
* as a result of not being hierarchal, it is not straightforward to undo changes if you end up with, for instance, two near-identical examplars. This could be implemented, but isn't implemented here.
* slows down as the number of clusters grows, not the best way to decluster all the way back to constituent members
* can only use Euclidian distance; CIELab and other color-distance metrics (which aren't all they're cracked up to be anyway) will not work (that's a tradeoff for some of the optimizations)

This clustering is designed for questions like "what are the two dominant colors in this image (respecting transparency)?"

Once you split a cluster, it's acceptable to just throw it away. And you will probably only need the exemplars of the clusters you do keep, though you could re-cluster each cluster's members for a semi-hierarchal clustering scheme.

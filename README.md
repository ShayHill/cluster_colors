# Cluster Colors (or other vectors)

Processing and clustering colors from images presents some challenges:

* Even a small (800x600) image will have up to 480,000 colors.
* Solutions like PIL's `Image.quantize`, on the other hand, make the colors sample *too* coarse.
* Even after reducing color variety, you're still dealing with 480,000 color *instances*.
* Solutions like `scikit-learn`'s `KMeans` handle some of these challenges, but are non-deterministic and not flexible in the ways that I'd like.

## I provide three steps here:

### Pool colors

Average similar colors. Specifically, this maps an 8-bit color space to an n-bit color space then averages colors in each bin. An argument, `nbits`, specifies the number of bits to use for each color channel. The default is 6, which reduces 17-million-ish possible colors to 300-thousand-ish possible colors. The downside is that the boundaries between n-bit bins are arbitrary. Heavy concentrations of near-identical colors will be split if a boundary passes through them.

Pooling colors from an image path will write a cache to your temp directory.

### Cut  colors

Reduce the number of colors by recursively splitting the color space along the longest axis. This is a median cut algorithm, but it's not constrained to x, y, or z axes. The longest axis is determined by the standard deviation of the colors in the cluster. I've made the cut just a little bit smarter than standard median cut, but this is essentially k-medoids without the re-distribution step, so it's more efficient, but not the best we can do. An argument, `num`, specifies the number of colors to reduce to. 512 is a good number, but if you're still missing some nuance, you can increase it.

### Divisive and Agglomerative clustering

* Both are deterministic.
* Both handle frequency, weight, transparency.
* Both allow a user-defined proximity matrix, so you can use whatever delta function you like as long as `delta(a,a)` is 0 and `delta(a,b)` is never 0. Common choices are Euclidean, squared Euclidean, and delta-e.
* Divisive uses a variation of median cut followed by a kmediods-like reassignment step to conversion.
* Agglomerative uses *complete linkage*.
* Divisive is more robust to outliers and will give more even-sized clusters.
* Divisive child clusters will not necessarily contain (or only contain) the members of the parent.
* Agglomerative is more likely to separate outliers.
* Agglomerative is heirarchical.

Divisive clustering is typically better for, "What are the five dominant colors in this image?"

Agglomerative clustering is typically better for, "How many colors do I need to represent this image with no more than `delta==3` between any two cluster members?"

## Installation

    pip install cluster_colors

## Basic usage

~~~python
from cluster_colors import get_image_clusters

# find the five most dominant colors in an image
clusters = get_image_clusters(image_filename)
clusters.split_to_n(5)
exemplars = clusters.get_as_vectors()

# to save the cluster exemplars as an image file
show_clusters(split_clusters, "open_file_to_see_clusters")
~~~

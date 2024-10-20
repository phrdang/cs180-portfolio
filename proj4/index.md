<!-- Mathjax Support -->
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Project 4

\[Auto\]Stitching Photo Mosaics - [Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj4/index.html)

1. Table of Contents
{:toc}

## Project 4A

Image Warping and Mosaicing - [Project Spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj4/partA.html)

### Shoot the Pictures

To do this project, I needed to take pictures to perform image rectification and mosaicing.

For the image rectification, I chose to take a picture of a [Computer Science Mentors](https://csmentors.studentorg.berkeley.edu/) CS 88 worksheet and a piece of art in Soda Hall 380 (both taken at an angle so they can later be rectified). I then [rescaled](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.rescale) the images by a factor of `0.3` to make processing the images faster in later parts (fewer pixels to operate on).

| Worksheet | Art |
| :--- | :--- |
| ![worksheet](assets/a/1/worksheet/worksheet.jpg) | ![art](assets/a/1/art/art.jpg) |

For image mosaicing, I needed to take 3 sets of photos of the same scenery and the same center of projection (e.g. only the camera lens rotates, but the axis of rotation is the same). I chose to take pictures of a hike on the [Berkeley Fire Trails](https://maps.app.goo.gl/qPmW1P4tjoZkJzYS8), a [path on campus](https://maps.app.goo.gl/k5gq2NBvM3SLgfNg7) between Valley Life Sciences Building and Haviland Hall, and a view of Doe Library and the Memorial Glade going down [the North Gate path](https://maps.app.goo.gl/rxjzH346awgBhQjA7). I also rescaled these images by a factor of `0.3`.

| Location | Left Image | Right Image |
| :--- | :--- | :--- |
| Fire Trails | ![fire trails left](assets/a/2/fire-trails/fire1.jpg) | ![fire trails right](assets/a/2/fire-trails/fire2.jpg) |
| Campus Path | ![campus path left](assets/a/2/campus/campus1.jpg) | ![campus path right](assets/a/2/campus/campus2.jpg) |
| Doe Library | ![doe library left](assets/a/2/doe-library/doe1.jpg) | ![doe library right](assets/a/2/doe-library/doe2.jpg) |

### Recover Homographies

A [homography](https://en.wikipedia.org/wiki/Homography) is a mapping between any 2 projective planes with the same center of projection. (See [lecture slides from Fall 2024](https://inst.eecs.berkeley.edu/~cs180/fa24/Lectures/mosaic.pdf#page=36).) We can use homographies to warp images and perform rectification and mosaicing.

To compute a homography from a source point $$(s_{x_i}, s_{y_i}, 1)$$ to a destination point $$(wd_{x_i}, wd_{y_i}, w)$$, you need to compute the values in the $$3 \times 3$$ homography matrix $$H$$ below. Also note that the source and destination points are [homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates).

$$
\begin{bmatrix}
a & b & c \\
d & e & f \\
g & h & 1
\end{bmatrix}
\begin{bmatrix}
s_{x_i} \\
s_{y_i} \\
1
\end{bmatrix}
=
\begin{bmatrix}
wd_{x_i} \\
wd_{y_i} \\
w
\end{bmatrix}
$$

Assuming you know $$H$$, you can apply it to every point $$i$$ in the source image.

To find $$a$$ through $$h$$, you need to solve this system of linear equations since we know $$s_x, s_y, d_x, d_y$$ for a subset of $$i$$, the correspondence points, which are manually marked using the [correspondence tool from Project 3](https://cal-cs180.github.io/fa23/hw/proj3/tool.html).

$$\begin{bmatrix}
s_{x_1} & s_{y_1} & 1 & 0 & 0 & 0 & -s_{x_1} * d_{x_1} & -s_{y_1} * d_{x_1} \\
0 & 0 & 0 & s_{x_1} & s_{y_1} & 1 & -s_{x_1} * d_{y_1} & -s_{y_1} * d_{y_1} \\
\dots & \dots & \dots & \dots & \dots & \dots & \dots & \dots \\
s_{x_n} & s_{y_n} & 1 & 0 & 0 & 0 & -s_{x_n} * d_{x_n} & -s_{y_n} * d_{x_n} \\
0 & 0 & 0 & s_{x_n} & s_{y_n} & 1 & -s_{x_n} * d_{y_n} & -s_{y_n} * d_{y_n}
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
d \\
e \\
f \\
g \\
h
\end{bmatrix}
=
\begin{bmatrix}
d_{x_1} \\
d_{y_1} \\
\dots \\
d_{x_n} \\
d_{y_n}
\end{bmatrix}
$$

Note: $$n$$ is the total number of homogeneous coordinate pairs in the source/destination image.

As you can see, the system is overdetermined if $$n > 4$$. Because of this, we must use [least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) to find a "best fit" solution.

### Warp the Images

Now that we have a way to compute $$H$$, we can perform warping by writing a function `warp_image(img, H)`. Here is an overview of the warping algorithm:

1. Compute $$H^{-1}$$
2. Determine the size of the warped image
    1. Get the $$(x, y, 1)$$ coordinates of the corners of the source image
    2. Warp the source corners to get the destination corners by doing `H @ src_corners`, where `src_corners` is a $$3 \times 4$$ matrix (each column is a homogeneous coordinate representing a corner)
    3. Normalize the destination corners (e.g. divide $$(wx, wy)$$ by $$w$$)
    4. Get the min and max $$x$$ and $$y$$ coordinates to figure out the size of the warped image
3. Determine all of the $$(x, y, 1)$$ coordinates inside the warped image. Call this $$3 \times n$$ matrix `dest_pts` (each column is a homogeneous coordinate).
4. Perform an inverse warp (like in [Project 3](../proj3/index.md))
    1. Do `H_inverse @ dest_pts`
    2. Normalize the matrix product like in step 2.3
    3. Use [`scipy.ndimage.map_coordinates`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates) to interpolate color values

### Image Rectification

To rectify the worksheet and art images, I marked their corners and then hardcoded their corresponding points based on the assumption that the worksheet is 8.5 x 11 inches and that the art has a 2:3 ratio (width:height):

| Source Points | Destination Points |
| :--- | :--- |
| ![worksheet source points](assets/a/1/worksheet/worksheet_pts.png) | ![worksheet destination points](assets/a/1/worksheet/worksheet_rectified_pts.png) |
| ![art source points](assets/a/1/art/art_pts.png) | ![art destination points](assets/a/1/art/art_rectified_pts.png) |

I then computed $$H_{\text{worksheet}}$$ and $$H_{\text{art}}$$, and performed the warping algorithm described in the previous section to rectify the images. I also cropped the resulting warped image to remove unnecessary black pixels created by performing the projective transformation.

| Worksheet Rectified | Art Rectified |
| :--- | :--- |
| ![worksheet rectified](assets/a/1/worksheet/worksheet_rectified.jpg) | ![art rectified](assets/a/1/art/art_rectified.jpg) |

Note that the rectified worksheet top is not perfectly straight despite the hardcoded rectangular destination points. This is because in the source image, the paper is not completely flat on the table due to the dog-eared corners.

### Blend the images into a mosaic

To create an image mosaic (e.g. stitching together each pair of images of the Berkeley landscape), I can also do the same warping using homographies. Specifically, the approach is to:

1. Determine correspondence points manually using the correspondence tool linked above
2. Warp image 1 to image 2
3. Zero pad warped image 1 and original image 2 so that their dimensions match
4. Blend warped image 1 with original image 2

I experimented with various blending methods. First, I tried a naive blending by taking the average of the padded images. This led to noticeable edges between warped image 1 and image 2:

<div style="text-align: center;">
<img src="assets/a/2/fire-trails/fire_combined.jpg" alt="fire trails blended using average blend" width="500">
</div>

Next I tried doing blending using a Laplacian stack with a "half half" alpha mask like in [Project 2](../proj2/index.md), where all the pixels on the left of the mask are 1 and all the pixels on the right of the mask are 0:

<div style="text-align: center;">
<img src="assets/a/2/half_half_mask.png" alt="half half alpha mask" width="500">
</div>

This was a significant improvement from the naive blending method, but there is a noticeable artifact at the top. You can also see a faint vertical line at the midpoint of the mosaic:

<div style="text-align: center;">
<img src="assets/a/2/fire-trails/fire_blended_half_half.png" alt="fire trails blended using half half alpha mask" width="500">
</div>

The best result was achieved with a mask using the [distance transform](https://en.wikipedia.org/wiki/Distance_transform) of each image using [cv2.distanceTransform](https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html), and then finding locations where left distance transform is greater than the right distance transform (I called this `where_greater`). The final mask is made by `np.dstack`-ing `where_greater` for each of the 3 color channels (RGB). I also cropped the blended images to remove any unnecessary black pixels.

| Image | Left Image Distance Transform | Right Image Distance Transform | Distance Transform 1 > Distance Transform 2 |
| :--- | :--- | :--- | :--- |
| Fire Trails | ![left fire trails image distance transform](assets/a/2/fire-trails/fire1_dist_transform.jpg) | ![right fire trails image distance transform](assets/a/2/fire-trails/fire2_dist_transform.jpg) | ![fire trails distance transform 1 greater than distance transform 2 visualization](assets/a/2/fire-trails/fire_where_greater.jpg) |
| Campus Path | ![left campus path image distance transform](assets/a/2/campus/campus1_dist_transform.jpg) | ![right campus path image distance transform](assets/a/2/campus/campus2_dist_transform.jpg) | ![campus path distance transform 1 greater than distance transform 2 visualization](assets/a/2/campus/campus_where_greater.jpg) |
| Doe Library | ![left doe library image distance transform](assets/a/2/doe-library/doe1_dist_transform.jpg) | ![right doe library image distance transform](assets/a/2/doe-library/doe2_dist_transform.jpg) | ![doe library distance transform 1 greater than distance transform 2 visualization](assets/a/2/doe-library/doe_where_greater.jpg) |

Here are the final blended results:

| Fire Trails | Campus Path | Doe Library |
| :--- | :--- | :--- |
| ![blended fire trails](assets/a/2/fire-trails/fire_blended.jpg) | ![blended campus path](assets/a/2/campus/campus_blended.jpg) | ![blended doe library](assets/a/2/doe-library/doe_blended.jpg) |

This entire mosaicing process can be further generalized to make a mosaic of multiple images from the same scenery to form a panorama. Instead of warping one image to another, we can warp all images to a center image. This will be left for the next part of the project!

## Project 4B

In progress!

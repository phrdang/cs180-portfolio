# Project 1

Images of the Russian Empire: Colorizing the [Prokudin-Gorskii photo collection](https://www.loc.gov/collections/prokudin-gorskii/)

## Introduction

The Prokudin-Gorskii photo collection is a collection of photos taken by [Sergey Prokudin-Gorskii](https://en.wikipedia.org/wiki/Sergey_Prokudin-Gorsky) before there was color photography. He took identical photos using red, green, and blue filters and imagined that in the future, there would be a method to combine all 3 channels into a single, color photo.

In this project, I was tasked with combining each RGB channel into a color photo using computational methods (primarily using  the `numpy` and `skimage` libraries). The main challenge in combining the images is aligning them on top of each other.

[Full project spec from Fall 2024](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj1/)

## Single Scale Alignment

For images with small dimensions, it is sufficient to align each channel by trying every possible translation within a search space of [-15, 15] pixels. For each translation, we calculate the similarity between the translated image (the image we want to align) and the base image (the image we want to align *to*) using some metric.

I experimented with 4 metrics:

1. Sum squared difference (SSD)
2. Euclidean distance
3. Normalized cross-correlation (NCC)
4. [Structural similarity](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)

Because they all had similar results for single scale alignment, I arbitrarily chose SSD as my metric for both single scale and multi-scale alignment.

Before aligning the images, I also cropped all image channels so that only the inner 80% of the image remains (e.g. 10% is cropped off each end of the height and width).

Below are the results for the 3 small `.jpg` images (`cathedral`, `monastery`, and `tobolsk`):

![Colorized cathedral](assets/cathedral.jpg)
![Colorized monastery](assets/monastery.jpg)
![Colorized Tobolsk](assets/tobolsk.jpg)

## Multi-Scale Alignment (Image Pyramid)

For images with larger dimensions, it would take too long to do an exhaustive search on the original image. To solve this problem, I used an [image pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing)). Specifically this was the algorithm I used:

1. Read in the RGB channels the same way as single scale alignment
2. Crop the channels the same way as single scale alignment
3. Align the R and G channels to the B channel using an image pyramid:
    1. Rescale the original images by a factor of 1/16
    2. Use single scale alignment on the rescaled images over a search space of [-15, 15] pixels to find the best displacement
    3. Rescale the rescaled image by a factor of 2
    4. Use single scale alignment on the rescaled image over a search space of [-2, 2] pixels (centered at the current best displacement) to find the best displacement
    5. Repeat steps 3.3 - 3.4 until the rescaled image reaches the same size as the original image
4. Use the final optimal displacement to translate the original image
5. `np.dstack` the aligned RGB channels to form the final colorized image the same way as single scale alignment

Below are the results for the large `.tif` images:

<table>
    <tr>
        <td><img src="assets/church.jpg" height="50%"></td>
        <td><img src="assets/emir.jpg" height="50%"></td>
        <td><img src="assets/harvesters.jpg" height="50%"></td>
    </tr>
    <tr>
        <td><img src="assets/icon.jpg" height="50%"></td>
        <td><img src="assets/lady.jpg" height="50%"></td>
        <td><img src="assets/melons.jpg" height="50%"></td>
    </tr>
    <tr>
        <td><img src="assets/onion_church.jpg" height="50%"></td>
        <td><img src="assets/sculpture.jpg" height="50%"></td>
        <td><img src="assets/self_portrait.jpg" height="50%"></td>
    </tr>
    <tr>
        <td><img src="assets/three_generations.jpg" height="50%"></td>
        <td><img src="assets/train.jpg" height="50%"></td>
        <td></td>
    </tr>
</table>

## Bells and Whistles (Extra Credit)

In construction!

## References

- [Alec Li's Project 1](https://inst.eecs.berkeley.edu/~cs180/fa23/upload/files/proj1/alec.li/)
- [Jeffrey Tan's Project 1](https://inst.eecs.berkeley.edu/~cs180/fa23/upload/files/proj1/tanjeffreyz02/)
- [Bryan Li's Project 1](https://inst.eecs.berkeley.edu/~cs180/fa23/upload/files/proj1/bryanli0/)
- [Minseok Son's Project 1](https://inst.eecs.berkeley.edu/~cs180/fa23/upload/files/proj1/tom5079/)

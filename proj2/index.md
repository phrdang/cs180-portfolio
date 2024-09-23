# Project 2

Fun with Filters and Frequencies

[Original project spec](https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj2/index.html)

## Part 1: Fun with filters

### Part 1.1: Finite Difference Operator

By convolving the original image with [finite difference](https://en.wikipedia.org/wiki/Finite_difference) operators,
I can obtain the gradient/magnitude image, which shows the edges of the image. To greater accentuate the edges,
I binarized the gradient/magnitude image (e.g. all pixels above a certain threshold are set to white and everything
else is black).

<table>
    <tr>
        <td>Original</td>
        <td>Partial Derivative (dx)</td>
        <td>Partial Derivative (dy)</td>
        <td>Gradient/Magnitude (Edges)</td>
        <td>Binarized</td>
    </tr>
    <tr>
        <td><img src="assets/part1/1/cameraman.jpg"></td>
        <td><img src="assets/part1/1/cameraman_dx.jpg"></td>
        <td><img src="assets/part1/1/cameraman_dy.jpg"></td>
        <td><img src="assets/part1/1/cameraman_magnitude.jpg"></td>
        <td><img src="assets/part1/1/cameraman_binarized.jpg"></td>
    </tr>
</table>

### Part 1.2: Derivative of Gaussian (DoG) Filter

To reduce noise on the edge detection, we can first blur the cameraman image using a [Gaussian filter](https://en.wikipedia.org/wiki/Gaussian_blur), and then convolve the blurred image with the finite difference operators.

<table>
    <tr>
        <td>Blurred</td>
        <td>Partial Derivative (dx)</td>
        <td>Partial Derivative (dy)</td>
        <td>Gradient/Magnitude (Edges)</td>
        <td>Binarized</td>
    </tr>
    <tr>
        <td><img src="assets/part1/2/blurred_cameraman.jpg"></td>
        <td><img src="assets/part1/2/blurred_dx.jpg"></td>
        <td><img src="assets/part1/2/blurred_dy.jpg"></td>
        <td><img src="assets/part1/2/blurred_magnitude.jpg"></td>
        <td><img src="assets/part1/2/blurred_binarized.jpg"></td>
    </tr>
</table>

We can also demonstrate the commutativity of convolving and applying the Gaussian filter. Below are the results of convolving the Gaussian with the finite difference operators, and then convolving the blurred finite difference operators with the original image.

<table>
    <tr>
        <td>DoG dx finite difference operator</td>
        <td>DoG dy finite difference operator</td>
        <td>DoG dx convolved with cameraman</td>
        <td>DoG dy convolved with cameraman</td>
        <td>DoG Magnitude</td>
        <td>DoG Binarized</td>
    </tr>
    <tr>
        <td><img src="assets/part1/2/dog_dx.jpg" height="100" width="auto"></td>
        <td><img src="assets/part1/2/dog_dy.jpg" height="100" width="auto"></td>
        <td><img src="assets/part1/2/dog_dx_cameraman.jpg"></td>
        <td><img src="assets/part1/2/dog_dy_cameraman.jpg"></td>
        <td><img src="assets/part1/2/dog_magnitude.jpg"></td>
        <td><img src="assets/part1/2/dog_binarized.jpg"></td>
    </tr>
</table>

## Part 2: Fun with Frequencies

### Part 2.1: Image "Sharpening"

To "sharpen" an image, you can take the original image and subtract the Gaussian blurred image to get the high frequencies of the image. Then, you can add the high frequencies to the original image. With some mathematical manipulation, you can perform the sharpening in one convolution.

<table>
    <tr>
        <td>Original</td>
        <td>Sharpened</td>
    </tr>
    <tr>
        <td><img src="assets/part2/1/taj.jpg"></td>
        <td><img src="assets/part2/1/taj_sharpened.jpg"></td>
    </tr>
    <tr>
        <td><img src="assets/part2/1/landscape.jpg"></td>
        <td><img src="assets/part2/1/landscape_sharpened.jpg"></td>
    </tr>
</table>

I also took an image, Gaussian blurred it, and then sharpened it. As you can see, the sharpening does not exactly undo the blur -- there is some information loss caused by the blur that isn't recovered by sharpening.

<table>
    <tr>
        <td>Original</td>
        <td>Blurred</td>
        <td>Sharpened</td>
    </tr>
    <tr>
        <td><img src="assets/part2/1/canoe.jpg"></td>
        <td><img src="assets/part2/1/canoe_blurred.jpg"></td>
        <td><img src="assets/part2/1/canoe_sharpened.jpg"></td>
    </tr>
</table>

### Part 2.2: Hybrid Images

### Part 2.3: Gaussian and Laplacian Stacks

### Part 2.4: Multiresolution Blending


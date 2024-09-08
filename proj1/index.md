# Project 1

Images of the Russian Empire: Colorizing the [Prokudin-Gorskii photo collection](https://www.loc.gov/collections/prokudin-gorskii/)

## Introduction

In construction!

## Single Scale Alignment

Explanation in construction!

- Search space: [-15, 15]
- Image similarity metric used: SSD (sum squared difference) of the 2 image matrices
- Cropped all image channels so that only the inner 80% of the image remains (e.g. 10% is cropped off each end of the height and width)

Below are the results for the 3 small `.jpg` images (`cathedral`, `monastery`, and `tobolsk`):

![Colorized cathedral](assets/cathedral.jpg)
![Colorized monastery](assets/monastery.jpg)
![Colorized Tobolsk](assets/tobolsk.jpg)

## Multi-Scale Alignment (Image Pyramid)

In construction!

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

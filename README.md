## A Comparative Performance Analysis of Feature Description Algorithms Implemented in OpenCV - The Framework
### Nels Numan, s1459929, Bachelor Thesis
###### Leiden University

This repository contains the framework used for the Bachelor Thesis *A Comparative Performance Analysis of Feature Description Algorithms Implemented in OpenCV* by Nels Numan. The framework has been forked adapted from [an evaluation framework created by GitHub user BloodAxe](https://github.com/BloodAxe/OpenCV-Features-Comparison).

### Compile instructions
The following libraries are required to have installed: OpenCV (version 3.3.0), Boost (minimum version 1.60.0). An optional libary is OpenMP.

1. Open your command-line
2. Clone the repository
``git clone git@github.com:nsalminen/OpenCV-Features-Comparison.git``
3. Open `CMakeLists.txt` and edit your *OpenCV directory*, *CMAKE_INCLUDE_PATH*, *CMAKE_LIBRARY_PATH* to link the appropriate library files.
4. Execute `cmake .`
5. Execute `make`

### Usage instructions
After compiling the framework, execute the program as following:

`./EvalFramework Source`

Where *Source* is the source folder of the images to be evaluated.

### Source Dataset Download
[Dataset link download (2500 images from the MIR Flickr Dataset)](https://dl.dropboxusercontent.com/u/49159172/dataset.tar.gz)
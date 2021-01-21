# Cell Signal Analysis

Code to identify the major cellular signals present in bulk transcriptomes (typically RNA-seq) where the input signals are usually defined using single cell transcriptomes.  This is conceptually similar to deconvolution of bulk RNA-seq using scRNA-seq as a reference and this code can be used in that way.  However, as described in [this paper](https://www.biorxiv.org/content/10.1101/2020.03.19.998815v2) this method has been designed to solve a different problem.  Specifically, the intended use case is where the reference set of cellular signals are incomplete or different in some way from the bulk transcriptomes to which it is applied.  An example of this is trying to model cancer tissues using the normal tissue from which it is derived.  In this case, the reference is known to be incomplete (cancer cells are different from the normal cell they derived from) and the intent is to prevent or mitigate the misallocation of inappropriate reference populations where they are not present.

## Installation

Although the core modelling is done using TensorFlow in python, the expectation is that this code is called from R using the provided functions.  The R prerequisites are minimal, but python3 and TensorFlow (v2) need to be installed.

## Usage

The data needs to be in a particular format described in the documentation at the top of the python code. I’m aware that getting the bulk data into the format required can be a bit of a pain, but it makes a considerable difference to the running time of the code once you are looking at data-sets with 100s or 1000s of samples (e.g. TCGA).

There is an example dataset that can be used to test if the software is installed correctly.  When run, it should produce a heatmap like exampleHeatmap.png included in this repository.

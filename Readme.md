# Cell Signal Analysis

Takes a reference single cell data set and attempts to explain bulk transcriptomes as a linear combination of "cellular signals", typically defined from populations present in reference single cell transcriptomes.  This is conceptually similar to deconvolution of bulk RNA-seq using scRNA-seq as a reference and this code can be used in that way.  However, as described in [this paper](https://www.biorxiv.org/content/10.1101/2020.03.19.998815v2) this method has been designed to solve a different problem.  Specifically, the intended use case is where the reference set of cellular signals are incomplete or different in some way from the bulk transcriptomes to which it is applied.  An example of this is trying to model cancer tissues using the normal tissue from which it is derived.  In this case, the reference is known to be incomplete (cancer cells are different from the normal cell they derived from) and the intent is to prevent or mitigate the misallocation of inappropriate reference populations where they are not present.

## Installation

Although the core modelling is done using TensorFlow in python, the expectation is that this code is called from R using the provided functions.  The R prerequisites are minimal, but various python packages need to be installed.  If you install tensorflow, pandas, numpy and scipy that should be enough.

## Usage

The data needs to be in a particular format described in the documentation at the top of the python code. I’m aware that getting the bulk data into the format required can be a bit of a pain, but it makes a considerable difference to the running time of the code once you are looking at data-sets with 100s or 1000s of samples (e.g. TCGA).

There is an example dataset that can be used to test if the software is installed correctly.  When run, it should produce a heatmap like exampleHeatmap.png included in this repository.

The ideal way to provide bulk data is to have one sample per file.  This seems cumbersome but makes things much faster when you have a TCGA size data set and only want to load 10 samples.  If you just have 10 samples in one giant TCGA sized matrix, the whole file must be loaded into memory, then the samples you want selected.  This requires lots of memory and slows things done immensely.  Each individual file should look like this:
        GeneName    GeneLength  SampleID
        CD13    1210    3
        CD4 4320    0
        EPCAM1  399 2
        ...

That is, it should be tab delimited, the first row must give column headers, the first column must give the names of genes and the second column must give the gene length.

If gene names differ between files (e.g. different bulk sample files are created using different annotations), only the global intersection of gene names is considered.  That is, any information pertaining to a gene not present in all samples and the reference is not considered and thrown out.  So take care not to use wildly discrepant annotations.

The output consists of:

        *_usedBulkCounts.tsv - The exact table of fragment counts used to do the fit.
        
        *_usedBulkGeneLengths.tsv - The exact table of per-sample gene lengths used to do the fit.
        
        *_usedCellularSignals.tsv - The definition of the cellular signals used to perform the fit.
        
        *_usedGeneWeights.tsv - The exact gene weights used to perform the fit.
        
        *_fitExposures.tsv - The main results file.  Fitted values for the relative contribution of the cellular signals and intercept term, for the samples provided.  There are also some goodness of fit metrics included in this table.  pR2, which is a pseudo-R-squared value, obsCount, which is the total number of fragment counts observed for each sample, and fitCount, which is the total fitted fragment count for each sample.
        
        *_fitCounts.tsv - The full fitted table of counts.
        
        *_negativeLogLikelihoodFullFit.tsv - This table gives the contribution to the total negative log-likelihood for each gene in each sample.  This is most useful for assesing which genes are most poorly explained by the model fit.  This log-likelihood is for the full model fit consisting of all celullalar signals and an intercept term.
        
        *_negativeLogLikelihoodNullFit.tsv - The same as above, but for the Null model where only an intercept term is used.

## Test run

Download the code and run the test.

```
git clone https://github.com/constantAmateur/cellSignalAnalysis.git
cd cellSignalAnalysis
mkdir output
python ./cellSignalAnalysis.py -b ./bulkData/SangerProjectTARGET_ALL_phase1.txt -s ./scData/PBMCRef -w ./geneWeights.tsv ./output/TestRun
```

Load and visualise the results in R.

```R
source('cellSignalAnalysis.R')
fit = normaliseExposures('./output/TestRun_fitExposures.tsv')
pdf('output/TestRun_plot.pdf',width=10,height=7)
hm = plotExposures(fit)
draw(hm)
dev.off()
```





The following assumes that you have conda and walks you through creating the required conda environment. Please note that since RDkit is not guaranteed to run on the latest version of python, our suggestion is that you use python 3.6 or earlier. 

1. Create and activate a new python environment and install pip:
conda create --name p36_repurposing python=3.6 pip
conda activate p36_repurposing

2. Install using pip required packages. Note that versioned requirements are being used here but any recent version should work:
pip install -r requirements.txt

3. Install RDkit from the RDkit channel
conda install -c rdkit rdkit

You may additionally install R with ggplot2 and ggthemes in order to use 'make_error_correlation_plots.R' which will give a comparison of performance on test sets.

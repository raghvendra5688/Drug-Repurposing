The following assumes that you have conda and walks you through creating the required conda environment. You can install conda from the following link (https://docs.anaconda.com/anaconda/install/). 
Please note that since RDkit is not guaranteed to run on the latest version of python, our suggestion is that you use python 3.6 or earlier. 

1. Create and activate a new python environment and install pip:
````
conda create --name p36_repurposing python=3.6 pip
````
````
conda activate p36_repurposing
````

2. Install torch using conda
````
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
````

3. Install using pip required packages. Note that versioned requirements are being used here but any recent version should work:
````
pip install -r requirements.txt --use-feature=2020-resolver
````

4. Install RDkit from the RDkit channel:
````
conda install -c rdkit rdkit
````

5. To install pytorch geometric with cuda do the following:
````
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
````


You may additionally install R with ggplot2 and ggthemes in order to use "_make_error_correlation_plots.R_" which will give a comparison of performance on test sets.

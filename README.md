# rna-localization



## Installation

1. Install conda

2. Perform `conda env create -n reggen -f environment.yml`
    - If you have already installed and activated a conda environment, **before you do the above command** deactivate the previous conda env with `conda deactivate`

3. Activate the conda environement with `conda activate reggen`

In case you are using a beautiful apple device with an M1 chip do the following **with the previously mentioned conda environment enabled**:

```
pip uninstall tensorflow
conda install -c apple tensorflow-deps
pip install tensorflow-macos

```

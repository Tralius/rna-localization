# rna-localization

## Background

In this repository we tackle the problem of rna-localization. In this repository we propose different models 
that predict the distribution of RNA sequences in cellular compartments.

## Setup

1. Install graphviz:
   - Please see https://graphviz.gitlab.io/download/ for that

2. Install conda (check if conda is already installed with `conda --version`, if not then get it from https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3. Perform `conda env create -n reggen -f environment.yml`
    - If you have already installed and activated a conda environment, **before you do the above command** deactivate the previous conda env with `conda deactivate`

4. Activate the conda environement with `conda activate reggen`

## Content

### Structure

In this section we describe the important repository content.

**dataloaders**: folder containing the our custom `GeneDataloader`. As described in the report, we allow the use of three different
inputs: sequence, struct, m6A. 

**model_architectures**: This folder contains architectures used during the past weeks. They come in form
_.yaml_ files and follow the writing convention found in the convention text files.

**test_data_evaluation**: This folder contains the notebooks used for predicting the labels of the test data set. 
There is a notebook for every trained model (seen in the paper). <br>

**models**: This folder contains the code for the used models. Interesting are the following scripts:

- **model.py, singlebranchModel.py, multibranchModel.py**: abstraction of models that can be constructed automatically from
the yaml files.
- **CNN_RNN_models.py**: final models created directly with the keras api.
- **utils.py**: script used to create and add the different needed layers specified in the architecture yaml file.
Possible layers: Convolution, Dense, Flatten, Drouput, MaxPooling, Attention, Skip, ...

**main**:

- notebooks: 

  - **singlebranch, multibranch**: templates for training pipelines of the specific model types
  - **attention_base, CNN_Baseline_4Conv_Struct_ext, dataPipeline_RNATracker**: notebooks used to train different models
  - **MotifSearch.ipynb**: notebooks used to generate the integrated gradients

    
- python scripts:

  - **plotting**: script containing different plotting modalities. Together with the **plot_res2.ipynb** they constitute
  the tools used to generate the plots portrayed in the report.
  - **utils**: functions used to load the datasets and prepare the parameters for the dataloaders and models.
  - **metrics**: custom Pearson metric for visualizing training with keras. This is the pearson correlation used to display
  the correlation between the ground truth and predicted compartments.


### Paper Results

The results present in the paper can be reproduced with the notebooks present under folder **test_data_evaluation**.
In order to run a notebook, follow these steps:

1. Copy the notebook under the parent folder (**rna-localization**)
2. Every notebook corresponds to a provided trained model (link can be found in the paper). Download the corresponding model.
3. Change the `path` variable in the notebook to the path of the model. Or change the path to the model accordingly.
4. Follow the steps in the notebook (Run the steps)

### Motif Search

The integrated gradients can be generated with the **MotifSearch.ipynb** notebook. In order to do it follow these steps:

1. Download the wanted model from the repository linked in the report.
2. Copy the model file in the rna-localization folder and change the value of the `model_name` variable in the notebook
3. Adjust the values of the `GeneDataloader` to correspond to the chosen inputs of the model.
4. Follow and run the steps in the notebook.

_Note_: Motif search was only tested with only sequence data as input. It might not work for models trained with multiple inputs.
We recommend using **Baseline_no_str** and **CNN_RNN_no_str**.

## References

### RNATracker
https://github.com/HarveyYan/RNATracker/blob/master/LICENSE

### DeepmRNALoc
https://github.com/Thales-research-institute/DeepmRNALoc
# Architecture Descriptions, Model Checkpoints, and Training Histories for NSGA-Net with PENGUIN on Protein Diffraction Data

This repository contains all the necessary scripts and data to reproduce figures and results from our ICPP 2023 paper **submission**.

## File Structure:

* `environments/`: contains two Conda environment files that contain all python package dependencies necessary

* `models/`: contains ZIP archives that include all packaged models generated from our experiments

* `icpp_training_results/`: contains tables for runs using the A4NN workflow and runs without on each GPU

* `protein_dataset/`: contains ZIP archives that have simulated protein diffraction patterns for each type of laser beam intensity. Used as datasets for "predict.py"

* `scripts/`: contains a Jupyter Notebook and a corresponding Python scriptthat allows for interactively reproducing figures from the paper _Composable Workflow for Accelerating Neural Architecture Search Using In Situ Analytics for Protein Classification_ . Also, includes Python script for using a specific model to predict on the dataset

```bash
├── environments
│   ├── cpu_environment.yml
│   └── gpu_environment.yml
├── icpp_training_results
│   ├── 1gpu
│   │   ├── no_stopping
│   │   │   ├── 1gpu_1e14_training_data.tab
│   │   │   ├── 1gpu_1e15_training_data.tab
│   │   │   └── 1gpu_1e16_training_data.tab
│   │   ├── stopping
│   │   │   ├── 1gpu_1e14_stopping_training_data.tab
│   │   │   ├── 1gpu_1e14_stopping_training_data.tab
│   │   │   └── 1gpu_1e14_stopping_training_data.tab
│   ├── 4gpu
│   │   ├── no_stopping
│   │   │   ├── 4gpu_1e14_training_data.tab
│   │   │   ├── 4gpu_1e15_training_data.tab
│   │   │   └── 4gpu_1e16_training_data.tab
│   │   ├── stopping
│   │   │   ├── 4gpu_1e14_stopping_training_data.tab
│   │   │   ├── 4gpu_1e14_stopping_training_data.tab
│   │   │   └── 4gpu_1e14_stopping_training_data.tab
├── models
│   ├── 1gpu
│   │   ├── a4nn
│   │   │   ├── gpu1_1e14_penguin.z01
│   │   │   ├── gpu1_1e14_penguin.z02
│   │   │   ├── gpu1_1e14_penguin.zip
│   │   │   ├── gpu1_1e15_penguin.z01
│   │   │   ├── gpu1_1e15_penguin.zip
│   │   │   ├── gpu1_1e16_penguin.z01
│   │   │   └── gpu1_1e16_penguin.zip
│   │   ├── no_a4nn
│   │   │   ├── gpu1_1e14_no_penguin.z01
│   │   │   ├── gpu1_1e14_no_penguin.z02
│   │   │   ├── gpu1_1e14_no_penguin.zip
│   │   │   ├── gpu1_1e15_no_penguin.z01
│   │   │   ├── gpu1_1e15_no_penguin.z02
│   │   │   ├── gpu1_1e15_no_penguin.zip
│   │   │   ├── gpu1_1e16_no_penguin.z01
│   │   │   └── gpu1_1e16_no_penguin.zip
│   ├── 4gpu
│   │   ├── a4nn
│   │   │   ├── gpu4_1e14_penguin.z01
│   │   │   ├── gpu4_1e14_penguin.zip
│   │   │   ├── gpu4_1e15_penguin.z01
│   │   │   ├── gpu4_1e15_penguin.zip
│   │   │   ├── gpu4_1e16_penguin.z01
│   │   │   └── gpu4_1e16_penguin.zip
│   │   ├── no_a4nn
│   │   │   ├── gpu4_1e14_no_penguin.z01
│   │   │   ├── gpu4_1e14_no_penguin.z02
│   │   │   ├── gpu4_1e14_no_penguin.zip
│   │   │   ├── gpu4_1e15_no_penguin.z01
│   │   │   ├── gpu4_1e15_no_penguin.z02
│   │   │   ├── gpu4_1e15_no_penguin.zip
│   │   │   ├── gpu4_1e16_no_penguin.z01
│   │   │   └── gpu4_1e16_no_penguin.zip
├── protein_dataset
│   ├── 1e14.zip.bz2
│   ├── 1e15.zip.bz2
│   └── 1e16.zip.bz2
├── scripts
│   ├── paper_analysis.ipynb
│   ├── paper_analysis.py
│   ├── PD_dataset.py
│   ├── predict.py
│   ├── time_to_run.tab
│   └── utils.py
└── README.txt
```


---
## Dependencies:

* To run the software in this Dataverse repository, you must have access to the Conda package management software on your local machine.
* Python package dependencies are included within the conda environemnt files included within this repository. More on this in the "Loading Conda Environments" section.

---
## Unpacking split dataset ZIP files:

The ZIP files that contain the results from our workflow executions are split according to a file size limit. To join the zip files together upon download of the dataset, please join the archives together
to create a complete archive before unzipping. Please run the following commands to properly unpack a ZIP archive and use the same format for other ZIP archives in this dataverse:

```
$ zip -FF gpu1_1e14_no_penguin.zip --out gpu1_1e14_no_penguin_FULL.zip
$ unzip -FF gpu1_1e14_no_penguin_FULL.zip
```

---
## Loading Conda Environments:

* If you have access to only CPUs, please load the "cpu_environment.yml" Conda environment with the following command: `conda env create -f cpu_environment.yml`


* If you have access to an NVIDIA GPU, please load the "gpu_environment.yml" Conda environment with the following command: `conda env create -f gpu_environment.yml`

---
## Reproducing Figures from ICPP 2023 Paper:

* To reproduce figures from the paper, either run an interactive session with the Jupyter Notebook `paper_analysis.ipynb` or execute the `paper_analysis.py` script. 
   * Use the python script if there is no access to Jupyter Notebooks on the local machine. To run `paper_analysis.py`, use the following command: `python paper_analysis.py`

   * All figures will be included in the `figures/` folder.

* To use a specific model to perform predictions on the dataset, run "predict.py". There are two arguments to include when running the script: `--model_path` and `--data_path`.
   * Example command to run script:
   ```
   $ python predict.py --model_path $HOME/pd_proj/classification-search-penguin-pd_8node_1gpuEach_1e15_class_noPenguin-macro-20230410-232830/arch_68/arch_68_epoch_24.pt --data_path $HOME/pd_proj/1e15/images/testset
   ```

* `scripts/predict.py` - Needs the following arguments:
   * `--model_path` - Location of the model file (ex. `gpu1/stopping/1e14/arch_99/arch_99_epoch_24.pt`)
   * `--data_path` - Location of dataset to predict on (ex. `protein_dataset/1e14/images/testset`)
   * `--out_path` (optional) - Location to put result files `predictions.csv` and `prediction_analysis.txt`. Default location is the `model_path`

---
## Additional Notes:

* Tables in tabular format (`.tab`) must be downloaded as `.csv` files in order for the scripts to be able to read them
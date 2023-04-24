# Architecture Descriptions, Model Checkpoints, and Training Histories for NSGA-Net with PENGUIN on Protein Diffraction Data


## File Structure:

* Environments
    * Two Conda environment files that contain all python package dependencies

* gpu1
    * Two directories for runs using the A4NN workflow and runs without 
    * Each directory contains ZIP files that include all models generated and saved from each runs

* gpu4
    * Two directories for runs using the A4NN workflow and runs without
    * Each directory contains ZIP files that include all models generated and saved from each run

* icpp_training_results
    * Two directories that have tables for runs using the A4NN workflow and runs without on each GPU

* protein_dataset
    * Three BZ2 ZIP files that have simulated protein diffraction patterns for each type of laser beam intensity

* scripts
    * Jupyter Notebook that allows for interactively reproducing figures from the paper _Composable Workflow for Accelerating Neural Architecture Search Using In Situ Analytics for Protein Classification_ 
        * Also a corresponding Python script that also does this in case Jupyter Notebook is not available on the device

   * Python script for using a specific model to predict on the dataset


---
## Notes

* `scripts/predict.py` - Needs the following arguments:
   * `--model_path` - Location of the model file (ex. `gpu1/stopping/1e14/arch_99/arch_99_epoch_24.pt`)
   * `--data_path` - Location of dataset to predict on (ex. `protein_dataset/1e14/images/testset`)
   * `--out_path` (optional) - Location to put result files `predictions.csv` and `prediction_analysis.txt`. Default location is the `model_path`


* Tables in tabular format (`.tab`) must be downloaded as `.csv` files in order for the scripts to be able to read them 
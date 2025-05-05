# Deep Unsupervised Anomaly Detection in High-Frequency Markets - JFDS
## source: https://zenodo.org/records/11048480

This is the code and data repository for the paper titled "Deep Unsupervised Anomaly Dectection in High-Frequency Markets" published in the Journal of Finance and Data Science.

## Executing the code
For a complete restart of the project, the user will have to sequentially run the scripts: 01_data_formatting.py, 02_data_preprocessing.py, 03_model_training.py, and 04_model_scoring.py, in that exact order. Note that this will lead to different results than the paper's.

To only replicate the results of the paper, the exact preprocessed dataset and trained models used are also included in the /data and /models folders respectively. The user simply has to execute the script 04_model_scoring.py.

## Main scripts
-01_data_formatting.py: formats the original, raw LOBSTER data set, and adds the synthesized frauds.

-02_data_preprocessing.py: preprocesses the formatted LOBSTER data to be used in detection models: feature computations, train/valid/test split, normalization.

-03_model_training.py: trains the detection models on the preprocessed LOBSTER data.

-04_model_scoring.py: scores the detection models on the preprocessed test data.

## Folders
This repository contains 5 folders: data, models, preprocessing, scoring, and train.

-data folder: contains the raw LOBSTER data set, its formatted version which includes the simulated frauds, and the preprocessed version to train and evaluate the models.

-models folder: contains the PyTorch models trained for this paper, along with their module.

-preprocessing folder: contains the utilities used for preprocessing the data.

-scoring folder: contains the utilities used for scoring the models.

-train folder: contains the utilities used to train the models.

## Data
The raw L1 LOBSTER data files, available at: https://lobsterdata.com/info/DataSamples.php, are compressed in the data.zip file. You need to extract this file to create the /data folder and access the dataset.

## Data Extraction
To extract the data files, please follow these steps:

1. **Extract data.zip to create the data folder**:
   - For Windows: Right-click on data.zip and select "Extract All", or use PowerShell: `Expand-Archive -Path data.zip -DestinationPath .`
   - For Linux/macOS: `unzip data.zip`

## Large Files Handling
This repository uses Git Large File Storage (Git LFS) to manage large files such as data.zip. To properly clone and work with this repository, please follow these steps:

1. **Install Git LFS**:
   - For Windows: `git lfs install`
   - For Linux/macOS: `git lfs install` (after installing Git LFS via package manager)

2. **Cloning the Repository**:
   ```
   git clone https://github.com/GarfieldHuang/stock-price-manipulation-detection.git
   cd stock-price-manipulation-detection
   git lfs pull
   ```

3. **After Cloning**:
   - Extract the data.zip file as described in the Data Extraction section
   - CSV and parquet files are tracked by Git LFS and can be used directly
   - No need to decompress any files



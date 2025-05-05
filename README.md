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

## 資料
原始的 L1 LOBSTER 資料檔案（可於 https://lobsterdata.com/info/DataSamples.php 取得）已壓縮在 data.zip 檔案中。您需要解壓縮此檔案以建立 /data 資料夾並存取資料集。

## 資料解壓縮
若要解壓縮資料檔案，請依照以下步驟操作：

1. **解壓縮 data.zip 以建立 data 資料夾**：
   - Windows：右鍵點擊 data.zip 並選擇「解壓縮全部」，或使用 PowerShell：`Expand-Archive -Path data.zip -DestinationPath .`
   - Linux/macOS：`unzip data.zip`

## 大型檔案處理
本專案使用 Git Large File Storage (Git LFS) 來管理大型檔案，如 data.zip。要正確複製和使用此儲存庫，請按照以下步驟操作：

1. **安裝 Git LFS**：
   - Windows： `git lfs install`
   - Linux/macOS： `git lfs install`（需先通過套件管理器安裝 Git LFS）

2. **複製儲存庫**：
   ```
   git clone https://github.com/GarfieldHuang/stock-price-manipulation-detection.git
   cd stock-price-manipulation-detection
   git lfs pull
   ```

3. **複製後操作**：
   - 按照「資料解壓縮」部分的說明解壓縮 data.zip 檔案
   - 所有必要的資料已包含在 data.zip 中
   - 解壓縮後即可開始使用



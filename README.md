# 高頻市場中的深度無監督異常檢測 - JFDS
# Deep Unsupervised Anomaly Detection in High-Frequency Markets - JFDS
## 資料來源 | Source: https://zenodo.org/records/11048480

這是「高頻市場中的深度無監督異常檢測」論文的程式碼和資料儲存庫，此論文發表於金融與資料科學期刊 (Journal of Finance and Data Science)。本專案使用深度學習方法檢測高頻交易市場中的異常行為，特別是操縱股價的行為模式。

This is the code and data repository for the paper titled "Deep Unsupervised Anomaly Detection in High-Frequency Markets" published in the Journal of Finance and Data Science.

> [繁體中文版 README](CHINESE_README.md) | [English README](ENGLISH_README.md)

## 執行程式碼
### 完整流程運行
若要從頭開始重新執行專案，使用者需要依照以下順序執行程式碼：
1. `01_data_formatting.py`：格式化原始資料
2. `02_data_preprocessing.py`：預處理資料
3. `03_model_training.py`：訓練模型
4. `04_model_scoring.py`：評估模型

註：此方式會產生與論文不同的結果，因為模型訓練過程中有隨機性。

### 僅重現論文結果
若只想重現論文中的結果，我們已在 `/data` 和 `/models` 資料夾中包含了確切的預處理資料集和訓練好的模型。使用者只需執行 `04_model_scoring.py` 程式碼即可。

### 環境設定
在執行任何程式碼前，請確保已安裝所有必要的相依套件：
```
pip install -r requirements.txt
```

## 主要程式碼檔案
- **01_data_formatting.py**: 處理原始 LOBSTER 資料集，並加入合成的詐欺模式。此程式讀取 `/data` 目錄中的原始訂單簿資料，格式化成統一格式，並標記異常交易模式。

- **02_data_preprocessing.py**: 預處理格式化後的 LOBSTER 資料，用於異常檢測模型。包含特徵計算、訓練/驗證/測試資料分割和標準化處理。此步驟生成的資料會儲存在 `/data/preprocessed_lobster` 目錄。

- **03_model_training.py**: 在預處理過的 LOBSTER 資料上訓練異常檢測模型。使用不同的深度學習架構（MLP 自動編碼器、堆疊式 LSTM 自動編碼器和 Transformer 自動編碼器）進行無監督學習，訓練完成的模型會儲存在 `/models` 目錄。

- **04_model_scoring.py**: 在預處理過的測試資料上評估異常檢測模型。計算各模型的準確率、召回率和 F1 分數，並生成 ROC 曲線來比較不同模型的性能。

## 資料夾結構
本專案包含 5 個主要資料夾：data、models、preprocessing、scoring 和 train。

- **data 資料夾**: 包含三種主要資料：
  - 原始 LOBSTER 資料集 (根目錄中的 CSV 檔案)
  - 格式化版本 (`/formatted_lobster/`)，包含模擬的異常交易
  - 預處理版本 (`/preprocessed_lobster/`)，用於訓練和評估模型
  - 中間處理資料 (`/intermediate_lobster/`)，包含特徵提取和標準化後的資料

- **models 資料夾**: 包含為本研究訓練的 PyTorch 模型，以及它們的定義模組。
  - `models.py`：定義了所有模型架構的程式碼
  - `best_MLPAutoencoder.pt`：訓練好的 MLP 自動編碼器模型
  - `best_StackedLSTMAutoencoder.pt`：訓練好的堆疊式 LSTM 自動編碼器模型
  - `best_TransformerAutoencoder.pt`：訓練好的 Transformer 自動編碼器模型

- **preprocessing 資料夾**: 包含用於資料預處理的工具函數。
  - `utilities.py`：處理資料清理、特徵工程和資料標準化的函數庫

- **scoring 資料夾**: 包含用於評分模型的工具函數。
  - `utilities.py`：計算模型性能指標和生成評估圖表的函數庫

- **train 資料夾**: 包含用於訓練模型的工具函數。
  - `utilities.py`：模型訓練、驗證和超參數調整的函數庫

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



#食物分類 Demo – ResNet18 + FastAPI

本專案使用 PyTorch 訓練 Food-101 資料集中前 20 種類別，並以 FastAPI 建立 RESTful API，可上傳食物圖片並回傳預測類別。

#專案亮點

- 使用 ResNet18 訓練 Food-101 Top-20 類別
- 完整訓練流程（資料處理、Top-N 篩選、標籤轉換）
- FastAPI 建立 `/predict` API，可接收圖片回傳分類
- 成功部署至 Render 平台，可線上測試

#專案結構

- `main.py`：FastAPI 主入口
- `model_utils.py`：模型載入與預測邏輯
- `class_names.py`：類別對應表（Top 20）
- `static/index.html`：前端頁面（未來可接 Vue 或表單）
- `requirements.txt`：依賴套件清單

#預計規劃

- 整合前端 HTML 或 Vue 互動頁面
- 顯示 Top-3 機率預測與圖示
- 加入卡路里估算與營養分析（延伸應用）

#Demo 
![image](https://github.com/user-attachments/assets/23f9ab77-b475-4ce9-9bf5-09a86f67290a)
Demo首頁
![image](https://github.com/user-attachments/assets/b64168d5-6009-471c-8d2b-d42ba989c3d7)
選擇食物圖片
![image](https://github.com/user-attachments/assets/40bc16db-5773-48ae-b235-17b3affcdd85)
雖然結果不盡人意，準確度不夠高，因為電腦記憶體不太夠跑，常常跑到過載。

#使用技術
- PyTorch
- torchvision / ResNet18
- FastAPI
- gdown / Google Drive
- Render 部署

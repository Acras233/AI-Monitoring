# 智能摄像监控平台

## 项目介绍

本项目是基于 Python + YOLOv8n + Moondream VLM模型 + SiliconFlow API 的智能摄像监控平台，旨在通过机器学习以及计算机视觉解决现实中存在的摄像安保问题。

### 主要功能

系统包含以下检测功能：

- **火焰检测**：实时检测画面中的火焰和烟雾
- **人员检测**：检测人员滞留行为
- **车辆检测**：检测车辆滞留行为
- **卷帘门检测**：检测卷帘门状态
- **围栏翻越检测**：检测围栏翻越行为

同时，系统能使用 Moondream 模型对画面进行实时分析。若设备性能允许，可以更换 Moondream 模型为更强大的其他模型。

### 技术栈

- **后端框架**：Flask
- **目标检测**：YOLOv8n (Ultralytics)
- **视觉语言模型**：Moondream VLM
- **深度学习框架**：PyTorch
- **图像处理**：OpenCV
- **API服务**：SiliconFlow API

---

## 部署步骤

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)

### 步骤一：下载模型文件

通过网盘分享的文件下载模型文件：

- **链接**：https://pan.baidu.com/s/1rHUUAAP6DRlLwcGVMv096w
- **提取码**：y5n5

下载完成后，将 `model.safetensors` 模型文件放在项目的 `Model` 文件夹中。

### 步骤二：配置API密钥

修改 `backend/config.py` 文件中的 SiliconFlow API 配置：

```python
SILICONFLOW_API_KEY = "Your API KEY"  # 替换为您自己的API密钥
```

您可以在 [SiliconFlow官网](https://siliconflow.cn/) 注册并获取API密钥。

### 步骤三：安装依赖

在项目根目录下执行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

### 步骤四：运行项目

完成以上配置后，运行以下命令启动服务：

```bash
python run.py
```

启动成功后，访问 http://localhost:5000 即可使用智能监控平台。

---

## 项目结构

```
AI Monitoring/
├── Model/                  # Moondream模型相关文件
│   ├── moondream.py
│   ├── vision.py
│   ├── config.py
│   └── ...
├── backend/                # 后端服务
│   ├── app.py             # Flask应用主入口
│   ├── config.py          # 配置文件
│   ├── detection_engine.py # 检测引擎
│   ├── moondream_api.py   # Moondream API接口
│   ├── siliconflow_api.py # SiliconFlow API接口
│   └── ...
├── frontend/               # 前端界面
│   ├── static/
│   └── templates/
├── data/                   # 数据存储目录
│   ├── monitoring.db      # 数据库
│   ├── warnings/          # 警告记录
│   ├── screenshots/       # 截图
│   └── uploads/           # 上传文件
├── requirements.txt        # 依赖列表
└── run.py                  # 启动脚本
```

---

## 注意事项

1. 首次运行时会自动创建所需的数据目录
2. 建议使用GPU以获得更好的检测性能
3. 如需更换更强的视觉语言模型，请参考 Moondream 相关文档进行配置

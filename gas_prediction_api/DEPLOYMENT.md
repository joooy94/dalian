# 天然气预测API部署指南 🚀

本文档提供天然气预测API服务的完整部署和运行指南。

## 📁 项目结构

```
gas_prediction_api/
├── .gitignore                          # Git忽略文件配置
├── API_DOCUMENTATION.md               # 详细API接口文档
├── README.md                          # 项目说明文档
├── DEPLOYMENT.md                      # 部署指南（本文档）
├── requirements.txt                   # Python依赖包列表
├── run.py                            # 简化启动脚本
├── api_gas_prediction_server.py      # 主要API服务器代码
├── example_client.py                 # 客户端使用示例
├── static/                           # 静态文件目录（图片）
└── stl_decomposition/                # 数据处理模块
    ├── __init__.py                   # Python包初始化文件
    └── api_data_fetcher.py          # API数据获取模块
```

## 🔧 环境要求

### 系统要求
- **操作系统**: Linux, macOS, Windows
- **Python版本**: Python 3.8 或更高版本
- **内存**: 建议 2GB 以上
- **磁盘空间**: 500MB 以上

### 软件依赖
- Python 3.8+
- pip 包管理器
- 网络连接（用于获取历史数据）

## 📦 快速部署

### 1. 环境准备

#### 方式1: 使用Pythån虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv gas_prediction_env

# 激活虚拟环境
# Linux/macOS:
source gas_prediction_env/bin/activate
# Windows:
gas_prediction_env\Scripts\activate
```

#### 方式2: 使用Conda环境（仅供开发测试时使用）

```bash
# 创建conda环境
conda create -n gas_prediction python=3.9
conda activate gas_prediction
```

### 2. 安装依赖

```bash
# 进入项目目录
cd gas_prediction_api

# 安装依赖包
pip install -r requirements.txt
```

### 3. 配置数据源

确保 `stl_decomposition/api_data_fetcher.py` 中的API配置正确：

```python
# 检查以下配置是否正确
url = "http://8.130.25.118:8000/api/hisdata"
```

### 4. 启动服务

#### 方式一：使用简化启动脚本（推荐）

```bash
python run.py
```

#### 方式二：直接运行主服务器

```bash
python api_gas_prediction_server.py
```

#### 方式三：使用uvicorn命令

```bash
uvicorn api_gas_prediction_server:app --host 127.0.0.1 --port 58888
```

### 5. 验证部署

访问以下URL验证服务是否正常：

- **服务主页**: http://127.0.0.1:58888/
- **健康检查**: http://127.0.0.1:58888/health
- **API文档**: http://127.0.0.1:58888/docs
- **交互式文档**: http://127.0.0.1:58888/redoc

## 🔧 高级配置

### 服务器配置

在 `api_gas_prediction_server.py` 或 `run.py` 中修改服务器设置：

```python
# 修改监听地址和端口
uvicorn.run(
    app,
    host='0.0.0.0',      # 监听所有网络接口
    port=8080,           # 自定义端口
    log_level='info',    # 日志级别
    workers=4            # 工作进程数（生产环境）
)
```

### 环境变量配置

创建 `.env` 文件来管理环境变量：

```bash
# .env 文件示例
API_HOST=127.0.0.1
API_PORT=58888
LOG_LEVEL=info
API_DATA_SOURCE_URL=your_api_url
```

### 预测参数调优

在 `APIGasPredictor` 类中调整预测参数：

```python
# 高斯滤波平滑参数
sigma = 3  # 增大值=更平滑，减小值=更敏感

# Holt-Winters 模型参数
seasonal_periods = 1440  # 季节性周期（分钟）

# 预测修正参数
bias_threshold = 0.12  # 偏差校正阈值
extreme_factor = 1.15  # 极值限制倍数
```

## 🚀 生产环境部署

### 使用Docker部署

1. **创建Dockerfile**：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 58888

# 启动命令
CMD ["python", "run.py"]
```

2. **构建和运行Docker镜像**：

```bash
# 构建镜像
docker build -t gas-prediction-api .

# 运行容器
docker run -d -p 58888:58888 --name gas-prediction gas-prediction-api
```

**部署版本**: 1.0.0  
**更新时间**: 2025年  
**维护状态**: 活跃维护 
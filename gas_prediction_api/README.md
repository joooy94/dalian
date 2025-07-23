# 天然气预测API服务 🚀

基于历史3周同期数据的天然气流量和压力预测服务，使用Holt-Winters指数平滑算法和FastAPI框架构建。

## 📋 项目概述

本项目提供天然气流量和压力的预测服务，通过分析历史3周同期数据，使用Holt-Winters指数平滑模型进行时间序列预测。API服务支持RESTful接口，提供流量预测、压力预测和可视化测试功能。

### 核心特性

- 🔮 **智能预测**：基于Holt-Winters指数平滑算法的时间序列预测
- 📊 **多指标支持**：同时支持天然气流量和压力预测
- 🕒 **高精度时间序列**：1分钟级别的预测精度
- 📈 **可视化测试**：提供预测结果对比图表
- 🚀 **高性能API**：基于FastAPI的异步RESTful服务
- 🔧 **智能修正**：自动偏差校正和极值处理

## 🏗️ 系统架构

```
天然气预测API服务
├── 数据获取层 (API Data Fetcher)
│   ├── 历史数据获取 (3周同期数据)
│   └── 数据验证和清洗
├── 数据处理层 (Data Processing)
│   ├── 高斯滤波平滑
│   ├── 时间序列对齐
│   └── 训练数据构建
├── 预测模型层 (Prediction Models)
│   ├── Holt-Winters指数平滑
│   ├── 季节性模式识别
│   └── 智能预测修正
├── API服务层 (FastAPI Server)
│   ├── 流量预测接口 (/predict)
│   ├── 压力预测接口 (/predict_pressure)
│   ├── 测试可视化接口 (/test)
│   └── 静态文件服务 (/static)
└── 可视化层 (Visualization)
    ├── 历史数据对比图
    └── 预测结果展示
```

## 🛠️ 技术栈

- **Python 3.8+**：核心编程语言
- **FastAPI**：现代异步Web框架
- **Pandas**：数据处理和分析
- **NumPy**：数值计算
- **SciPy**：科学计算（高斯滤波）
- **Statsmodels**：统计建模（Holt-Winters）
- **Matplotlib**：数据可视化
- **Pydantic**：数据验证和序列化
- **Uvicorn**：ASGI服务器

## 📦 安装与配置

### 环境要求

- Python 3.8+
- pip 或 conda 包管理器

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd dalian

# 安装依赖
pip install -r requirements.txt
```

### 配置数据源

确保 `stl_decomposition/api_data_fetcher.py` 文件正确配置了数据源API接口。

## 🚀 快速启动

### 启动服务器

```bash
python api_gas_prediction_server.py
```

服务器将在 `http://127.0.0.1:58888` 启动。

### 验证服务

访问以下URL验证服务是否正常运行：

- **服务信息**：http://127.0.0.1:58888/
- **健康检查**：http://127.0.0.1:58888/health
- **API文档**：http://127.0.0.1:58888/docs

## 📖 API接口文档

### 1. 流量预测接口

**端点**：`GET /predict`

**描述**：预测指定日期的天然气流量

**参数**：
- `date` (必需)：预测日期，格式为 `YYYY-MM-DD`

**示例请求**：
```bash
curl "http://127.0.0.1:58888/predict?date=2025-07-07"
```

**响应格式**：
```json
{
    "success": true,
    "prediction_date": "2025-07-07",
    "metric": "瞬时流量",
    "data_points": 1440,
    "predictions": [
        {
            "timestamp": "2025-07-07T00:00:00",
            "forecast": 123.45
        },
        {
            "timestamp": "2025-07-07T00:01:00",
            "forecast": 124.56
        }
        // ... 1440个数据点 (每分钟一个)
    ]
}
```

### 2. 压力预测接口

**端点**：`GET /predict_pressure`

**描述**：预测指定日期的天然气压力

**参数**：
- `date` (必需)：预测日期，格式为 `YYYY-MM-DD`

**示例请求**：
```bash
curl "http://127.0.0.1:58888/predict_pressure?date=2025-07-07"
```

**响应格式**：
```json
{
    "success": true,
    "prediction_date": "2025-07-07",
    "metric": "总压力",
    "data_points": 1440,
    "predictions": [
        {
            "timestamp": "2025-07-07T00:00:00",
            "forecast": 0.85
        },
        {
            "timestamp": "2025-07-07T00:01:00",
            "forecast": 0.86
        }
        // ... 1440个数据点
    ]
}
```

### 3. 测试预测接口

**端点**：`GET /test`

**描述**：生成预测结果并返回可视化图表

**参数**：
- `date` (必需)：预测日期，格式为 `YYYY-MM-DD`
- `metric` (可选)：预测指标，可选值为 `flow`（流量）或 `pressure`（压力），默认为 `flow`

**示例请求**：
```bash
# 测试流量预测
curl "http://127.0.0.1:58888/test?date=2025-07-07&metric=flow"

# 测试压力预测
curl "http://127.0.0.1:58888/test?date=2025-07-07&metric=pressure"
```

**响应格式**：
```json
{
    "success": true,
    "prediction_date": "2025-07-07",
    "metric": "瞬时流量",
    "plot_file": "test_flow_prediction_20250707.png",
    "message": "预测完成，共生成1440个数据点。可通过 /static/test_flow_prediction_20250707.png 查看预测图表。"
}
```

**查看图表**：
```
http://127.0.0.1:58888/static/test_flow_prediction_20250707.png
```

### 4. 静态文件服务

**端点**：`GET /static/{filename}`

**描述**：访问生成的图片文件

**示例**：
```
http://127.0.0.1:58888/static/test_flow_prediction_20250707.png
```

### 5. 健康检查接口

**端点**：`GET /health`

**描述**：检查服务健康状态

**响应格式**：
```json
{
    "status": "healthy",
    "service": "天然气预测API",
    "version": "1.0.0"
}
```

### 6. 服务信息接口

**端点**：`GET /`

**描述**：获取服务基本信息和可用端点

**响应格式**：
```json
{
    "service": "天然气预测API服务",
    "version": "1.0.0",
    "endpoints": {
        "flow_prediction": "/predict?date=YYYY-MM-DD",
        "pressure_prediction": "/predict_pressure?date=YYYY-MM-DD",
        "test_prediction": "/test?date=YYYY-MM-DD&metric=flow/pressure",
        "static_files": "/static/{filename}",
        "health_check": "/health"
    },
    "description": "基于历史3周同期数据的天然气流量和压力预测服务"
}
```

## 🔬 预测算法说明

### 数据获取策略

1. **历史数据选择**：获取预测日期前3周同一星期几的数据
   - 3周前同一天（如：预测周一，获取3周前的周一）
   - 2周前同一天
   - 1周前同一天

2. **数据验证**：确保每天至少有1000个数据点（约17小时的数据）

### 数据处理流程

1. **数据平滑**：使用高斯滤波（σ=3）减少噪声
2. **时间对齐**：将3周数据按时间顺序连接成训练序列
3. **质量检查**：处理缺失值和异常值

### Holt-Winters预测模型

1. **模型配置**：
   - 季节性周期：1440分钟（1天）
   - 季节性类型：加法季节性
   - 趋势：无趋势（更稳定）

2. **智能修正**：
   - **偏差校正**：检测预测均值偏差并自动校正
   - **负值处理**：将负值替换为合理最小值
   - **极值限制**：限制预测值在历史数据合理范围内
   - **平滑处理**：轻度滑动窗口平滑

## 📊 使用示例

### Python客户端示例

```python
import requests
import json
import pandas as pd

# 获取流量预测
response = requests.get('http://127.0.0.1:58888/predict?date=2025-07-07')
data = response.json()

if data['success']:
    # 转换为DataFrame
    df = pd.DataFrame(data['predictions'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"预测日期: {data['prediction_date']}")
    print(f"数据点数: {data['data_points']}")
    print(f"预测范围: {df['forecast'].min():.2f} ~ {df['forecast'].max():.2f}")
    print(f"预测均值: {df['forecast'].mean():.2f}")

# 获取测试图表
test_response = requests.get('http://127.0.0.1:58888/test?date=2025-07-07&metric=flow')
test_data = test_response.json()

if test_data['success']:
    print(f"可视化图表: http://127.0.0.1:58888/static/{test_data['plot_file']}")
```

### JavaScript客户端示例

```javascript
// 获取流量预测
async function getFlowPrediction(date) {
    try {
        const response = await fetch(`http://127.0.0.1:58888/predict?date=${date}`);
        const data = await response.json();
        
        if (data.success) {
            console.log(`预测日期: ${data.prediction_date}`);
            console.log(`数据点数: ${data.data_points}`);
            console.log('预测数据:', data.predictions);
            return data.predictions;
        }
    } catch (error) {
        console.error('预测请求失败:', error);
    }
}

// 使用示例
getFlowPrediction('2025-07-07');
```

## 🔧 配置说明

### 服务器配置

在 `api_gas_prediction_server.py` 文件末尾可以修改服务器配置：

```python
# 修改监听地址和端口
uvicorn.run(app, host='0.0.0.0', port=8000)
```

### 预测参数调优

可以在 `APIGasPredictor` 类中调整以下参数：

- **高斯滤波参数**：`sigma=3`（数据平滑程度）
- **季节性周期**：`seasonal_periods=1440`（分钟）
- **偏差校正阈值**：`data_std * 0.12`
- **极值限制倍数**：`data_max * 1.15`, `data_min * 0.85`

## 🐛 故障排除

### 常见问题

1. **数据获取失败**
   - 检查 `api_data_fetcher.py` 中的API配置
   - 确认网络连接和API服务可用性

2. **预测精度不理想**
   - 增加历史数据周数
   - 调整高斯滤波参数
   - 修改Holt-Winters模型参数

3. **服务启动失败**
   - 检查端口是否被占用
   - 验证所有依赖包是否正确安装

### 日志调试

服务运行时会输出详细日志，包括：
- 数据获取过程
- 预测模型训练
- 修正算法执行
- API请求处理
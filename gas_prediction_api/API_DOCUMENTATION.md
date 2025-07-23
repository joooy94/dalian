# 天然气预测API接口文档

**版本**: 1.0.0  
**基础URL**: `http://127.0.0.1:58888`  
**协议**: HTTP/HTTPS  
**数据格式**: JSON  

---

## 🌟 API概览

天然气预测API提供基于历史3周同期数据的时间序列预测服务，支持流量和压力两种指标的预测。API采用RESTful设计，所有接口均为GET请求，返回JSON格式数据。

### 核心功能
- 天然气流量预测 (1分钟级精度)
- 天然气压力预测 (1分钟级精度)  
- 预测结果可视化测试
- 健康状态监控
- 静态文件服务

---

## 📋 接口列表

| 接口 | 方法 | 端点 | 描述 |
|------|------|------|------|
| 流量预测 | GET | `/predict` | 预测指定日期的天然气流量 |
| 压力预测 | GET | `/predict_pressure` | 预测指定日期的天然气压力 |
| 测试预测 | GET | `/test` | 生成预测结果并返回可视化图表 |
| 静态文件 | GET | `/static/{filename}` | 访问生成的图片文件 |
| 健康检查 | GET | `/health` | 检查服务健康状态 |
| 服务信息 | GET | `/` | 获取服务基本信息 |

---

## 🔍 详细接口说明

### 1. 流量预测接口

#### 基本信息
- **端点**: `GET /predict`
- **描述**: 基于历史3周同期数据预测指定日期的天然气流量
- **认证**: 无需认证
- **限流**: 无限制

#### 请求参数

| 参数名 | 类型 | 必需 | 描述 | 示例 |
|--------|------|------|------|------|
| `date` | string | 是 | 预测日期，格式为YYYY-MM-DD | `2025-07-07` |

#### 请求示例

```bash
# cURL
curl "http://127.0.0.1:58888/predict?date=2025-07-07"

# Python requests
import requests
response = requests.get('http://127.0.0.1:58888/predict?date=2025-07-07')

# JavaScript fetch
fetch('http://127.0.0.1:58888/predict?date=2025-07-07')
```

#### 响应格式

**成功响应 (200 OK)**

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
        // ... 共1440个数据点 (每分钟一个)
    ]
}
```

**字段说明**

| 字段 | 类型 | 描述 |
|------|------|------|
| `success` | boolean | 请求是否成功 |
| `prediction_date` | string | 预测日期 |
| `metric` | string | 预测指标名称 |
| `data_points` | integer | 预测数据点数量 |
| `predictions` | array | 预测结果数组 |
| `predictions[].timestamp` | string | 时间戳 (ISO 8601格式) |
| `predictions[].forecast` | number | 预测值 |

**错误响应**

```json
// 400 Bad Request - 日期格式错误
{
    "detail": {
        "error": "日期格式错误",
        "message": "请使用YYYY-MM-DD格式"
    }
}

// 500 Internal Server Error - 预测失败
{
    "detail": {
        "error": "预测失败",
        "message": "无法获取足够的历史数据或预测过程出错"
    }
}
```

---

### 2. 压力预测接口

#### 基本信息
- **端点**: `GET /predict_pressure`
- **描述**: 基于历史3周同期数据预测指定日期的天然气压力
- **认证**: 无需认证
- **限流**: 无限制

#### 请求参数

| 参数名 | 类型 | 必需 | 描述 | 示例 |
|--------|------|------|------|------|
| `date` | string | 是 | 预测日期，格式为YYYY-MM-DD | `2025-07-07` |

#### 请求示例

```bash
curl "http://127.0.0.1:58888/predict_pressure?date=2025-07-07"
```

#### 响应格式

**成功响应 (200 OK)**

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
        // ... 共1440个数据点
    ]
}
```

---

### 3. 测试预测接口

#### 基本信息
- **端点**: `GET /test`
- **描述**: 生成预测结果并返回可视化图表，用于测试和验证预测效果
- **认证**: 无需认证
- **限流**: 无限制

#### 请求参数

| 参数名 | 类型 | 必需 | 描述 | 默认值 | 可选值 |
|--------|------|------|------|--------|--------|
| `date` | string | 是 | 预测日期，格式为YYYY-MM-DD | - | - |
| `metric` | string | 否 | 预测指标类型 | `flow` | `flow`, `pressure` |

#### 请求示例

```bash
# 测试流量预测
curl "http://127.0.0.1:58888/test?date=2025-07-07&metric=flow"

# 测试压力预测
curl "http://127.0.0.1:58888/test?date=2025-07-07&metric=pressure"

# 使用默认参数(流量)
curl "http://127.0.0.1:58888/test?date=2025-07-07"
```

#### 响应格式

**成功响应 (200 OK)**

```json
{
    "success": true,
    "prediction_date": "2025-07-07",
    "metric": "瞬时流量",
    "plot_file": "test_flow_prediction_20250707.png",
    "message": "预测完成，共生成1440个数据点。可通过 /static/test_flow_prediction_20250707.png 查看预测图表。"
}
```

**查看生成的图表**

```
GET http://127.0.0.1:58888/static/test_flow_prediction_20250707.png
```

---

### 4. 静态文件服务

#### 基本信息
- **端点**: `GET /static/{filename}`
- **描述**: 提供对生成的图片文件的访问
- **支持格式**: PNG, JPG, GIF等图片格式

#### 请求参数

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| `filename` | string | 是 | 文件名 |

#### 请求示例

```bash
curl "http://127.0.0.1:58888/static/test_flow_prediction_20250707.png" -o prediction.png
```

#### 响应格式

- **成功**: 返回文件内容 (Content-Type: image/png)
- **失败**: 404 Not Found

---

### 5. 健康检查接口

#### 基本信息
- **端点**: `GET /health`
- **描述**: 检查API服务健康状态
- **用途**: 服务监控、负载均衡器健康检查

#### 请求示例

```bash
curl "http://127.0.0.1:58888/health"
```

#### 响应格式

```json
{
    "status": "healthy",
    "service": "天然气预测API",
    "version": "1.0.0"
}
```

---

### 6. 服务信息接口

#### 基本信息
- **端点**: `GET /`
- **描述**: 获取API服务基本信息和可用端点列表

#### 响应格式

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

---

## 🔧 技术规格

### 数据格式规范

#### 时间戳格式
- **标准**: ISO 8601
- **格式**: `YYYY-MM-DDTHH:MM:SS`
- **时区**: 本地时区
- **示例**: `2025-07-07T14:30:00`

#### 预测值精度
- **流量**: 保留2位小数，单位根据数据源
- **压力**: 保留2位小数，单位根据数据源

#### 数据点数量
- **每日数据点**: 1440个 (每分钟一个)
- **预测时间范围**: 00:00:00 - 23:59:00

### 性能指标

| 指标 | 规格 |
|------|------|
| 响应时间 | < 120秒  |
| 并发请求 | 支持异步处理 |
| 数据精度 | 1分钟级别 |
| 预测周期 | 1天 (1440分钟) |
| 历史数据依赖 | 3周同期数据 |

### 错误处理

#### HTTP状态码

| 状态码 | 含义 | 场景 |
|--------|------|------|
| 200 | OK | 请求成功 |
| 400 | Bad Request | 参数错误 |
| 404 | Not Found | 资源不存在 |
| 500 | Internal Server Error | 服务器内部错误 |

#### 错误响应结构

```json
{
    "detail": {
        "error": "错误类型",
        "message": "详细错误信息"
    }
}
```

---

## 💡 使用最佳实践

### 1. 请求频率控制
- 建议每个预测日期只请求一次
- 可以缓存预测结果降低服务器负载
- 避免在短时间内重复请求相同数据

### 2. 错误处理建议

```python
import requests
import time

def get_prediction_with_retry(date, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(f'http://127.0.0.1:58888/predict?date={date}')
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            raise e
```

### 3. 数据处理建议

```python
import pandas as pd

def process_prediction_data(api_response):
    """处理API响应数据"""
    if not api_response.get('success'):
        raise ValueError("API请求失败")
    
    # 转换为DataFrame便于分析
    df = pd.DataFrame(api_response['predictions'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    return df
```

### 4. 可视化图表使用

```python
def get_prediction_chart(date, metric='flow'):
    """获取预测图表"""
    test_url = f'http://127.0.0.1:58888/test?date={date}&metric={metric}'
    response = requests.get(test_url)
    
    if response.status_code == 200:
        data = response.json()
        chart_url = f"http://127.0.0.1:58888/static/{data['plot_file']}"
        return chart_url
    return None
```

---

## 🔍 调试和监控

### 日志级别
- **INFO**: 正常请求处理
- **WARNING**: 数据质量问题
- **ERROR**: 预测失败或系统错误

### 监控建议
1. 定期调用 `/health` 接口监控服务状态
2. 监控响应时间和错误率
3. 检查预测结果的合理性

### 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 响应时间过长 | 历史数据获取慢 | 检查数据源API状态 |
| 预测值异常 | 历史数据质量差 | 检查输入日期的历史数据 |
| 500错误 | 服务内部错误 | 查看服务日志排查问题 |

---

## 🚀 SDK示例

### Python SDK示例

```python
class GasPredictionClient:
    def __init__(self, base_url='http://127.0.0.1:58888'):
        self.base_url = base_url
    
    def predict_flow(self, date):
        """预测流量"""
        url = f"{self.base_url}/predict?date={date}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def predict_pressure(self, date):
        """预测压力"""
        url = f"{self.base_url}/predict_pressure?date={date}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def test_prediction(self, date, metric='flow'):
        """测试预测并获取图表"""
        url = f"{self.base_url}/test?date={date}&metric={metric}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# 使用示例
client = GasPredictionClient()
flow_data = client.predict_flow('2025-07-07')
pressure_data = client.predict_pressure('2025-07-07')
```

### JavaScript SDK示例

```javascript
class GasPredictionClient {
    constructor(baseUrl = 'http://127.0.0.1:58888') {
        this.baseUrl = baseUrl;
    }
    
    async predictFlow(date) {
        const response = await fetch(`${this.baseUrl}/predict?date=${date}`);
        if (!response.ok) throw new Error('预测请求失败');
        return response.json();
    }
    
    async predictPressure(date) {
        const response = await fetch(`${this.baseUrl}/predict_pressure?date=${date}`);
        if (!response.ok) throw new Error('预测请求失败');
        return response.json();
    }
    
    async testPrediction(date, metric = 'flow') {
        const response = await fetch(`${this.baseUrl}/test?date=${date}&metric=${metric}`);
        if (!response.ok) throw new Error('测试请求失败');
        return response.json();
    }
}

// 使用示例
const client = new GasPredictionClient();
const flowData = await client.predictFlow('2025-07-07');
const pressureData = await client.predictPressure('2025-07-07');
```

---

## 📋 更新日志

### v1.0.0 (当前版本)
- ✅ 初始版本发布
- ✅ 流量和压力预测功能
- ✅ 可视化测试接口
- ✅ 基于FastAPI的高性能架构
- ✅ 完整的错误处理和日志记录

---

**文档更新时间**: 2025年  
**API版本**: 1.0.0  
**联系方式**: 通过GitHub Issues反馈问题 
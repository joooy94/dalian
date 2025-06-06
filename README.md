# SARIMA气体产量预测系统

本项目使用SARIMA (Seasonal AutoRegressive Integrated Moving Average) 模型对时间序列气体产量数据进行分析和预测。

## 项目结构

- `sarima_train.py` - 训练模块，用于训练SARIMA模型并保存
- `sarima_predict.py` - 预测模块，使用训练好的模型进行预测
- `sarima_95interval.py` - 完整版脚本，包含训练和预测，带95%置信区间

## 功能特点

- 数据预处理和可视化
- 自动划分训练集和测试集
- 预测结果包含95%置信区间
- 可以使用最近3天数据预测未来1天
- 支持指定预测日期
- 结果可视化

## 使用方法

### 训练模型

```bash
python sarima_train.py
```

这将使用默认参数训练模型，并将模型保存到`sarima_results`目录。

### 使用训练好的模型进行预测

```bash
python sarima_predict.py --input_file your_data.csv --input_days 3
```

参数说明:
- `--input_file`: 输入数据文件路径 (CSV格式，必须包含日期和目标列)
- `--date_column`: 日期列名 (默认: 'date')
- `--target_column`: 目标列名 (默认: 'OT')
- `--model_dir`: 模型目录 (默认: 'sarima_results')
- `--model_file`: 模型文件名 (默认: 'sarima_model.pkl')
- `--input_days`: 使用最近几天的数据进行预测 (默认: 3)
- `--prediction_date`: 预测日期 (格式: YYYY-MM-DD)，默认为输入数据的最后一天的下一天
- `--output_dir`: 输出目录 (默认: 'sarima_results')

### 示例: 使用最近3天数据预测未来1天

```bash
python sarima_predict.py --input_file data.csv --input_days 3
```

### 示例: 预测特定日期

```bash
python sarima_predict.py --input_file data.csv --prediction_date 2025-06-01
```

## 部署流程

1. **训练阶段**:
   ```bash
   python sarima_train.py
   ```
   这将生成模型文件 `sarima_results/sarima_model.pkl`

2. **预测阶段**:
   ```bash
   python sarima_predict.py --input_file recent_data.csv
   ```
   这将使用最近数据预测未来一天，并生成预测结果和图表

## 输出结果

预测脚本会生成以下输出:

1. 预测结果CSV文件，包含:
   - 预测值
   - 95%置信区间下界
   - 95%置信区间上界

2. 预测图表，显示:
   - 最近一天的历史数据
   - 未来一天的预测值
   - 95%置信区间

## 依赖库

- pandas
- numpy
- matplotlib
- statsmodels
- joblib

## 安装依赖

```bash
pip install pandas numpy matplotlib statsmodels joblib
``` 
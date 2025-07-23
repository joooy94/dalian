#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天然气预测API服务器
提供流量和压力预测的REST API接口
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime, timedelta
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 导入API数据获取模块
from stl_decomposition.api_data_fetcher import fetch_specific_days

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Gas_Prediction_Server")

# 创建FastAPI应用
app = FastAPI(
    title="天然气预测API服务",
    description="基于历史3周同期数据的天然气流量和压力预测服务",
    version="1.0.0"
)

# 数据模型
class PredictionPoint(BaseModel):
    timestamp: str
    forecast: float

class PredictionResponse(BaseModel):
    success: bool
    prediction_date: str
    metric: str
    data_points: int
    predictions: List[PredictionPoint]

class ErrorResponse(BaseModel):
    error: str
    message: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class ServiceInfo(BaseModel):
    service: str
    version: str
    endpoints: dict
    description: str

class TestResponse(BaseModel):
    success: bool
    prediction_date: str
    metric: str
    plot_file: str
    message: str

class APIGasPredictor:
    def __init__(self):
        """初始化API预测器"""
        self.raw_data = None
        self.smoothed_data_flow = None
        self.smoothed_data_pressure = None
        self.training_data_flow = None
        self.training_data_pressure = None
        
    def load_specific_dates_from_api(self, prediction_date):
        """
        从API加载预测所需的特定历史日期数据
        
        Args:
            prediction_date: 预测日期（字符串或datetime对象）
        """
        logger.info("正在从API加载预测所需的历史数据...")
        
        # 确定预测日期
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date).date()
        else:
            prediction_date = pd.to_datetime(prediction_date).date()
        
        # 获取预测日期的星期几
        prediction_weekday = pd.to_datetime(prediction_date).weekday()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        prediction_weekday_name = weekday_names[prediction_weekday]
        
        logger.info(f"预测日期: {prediction_date} ({prediction_weekday_name})")
        
        # 计算需要的历史数据日期（上周、上上周、上上上周的同一天）
        last_week_date = prediction_date - timedelta(days=7)  # 上周同一天
        two_weeks_ago_date = prediction_date - timedelta(days=14)  # 上上周同一天
        three_weeks_ago_date = prediction_date - timedelta(days=21)  # 上上上周同一天
        
        required_dates = [
            three_weeks_ago_date.strftime("%Y-%m-%d"),
            two_weeks_ago_date.strftime("%Y-%m-%d"),
            last_week_date.strftime("%Y-%m-%d")
        ]
        
        logger.info(f"需要获取的历史数据日期:")
        logger.info(f"  上上上周同一天: {required_dates[0]} ({prediction_weekday_name})")
        logger.info(f"  上上周同一天: {required_dates[1]} ({prediction_weekday_name})")
        logger.info(f"  上周同一天: {required_dates[2]} ({prediction_weekday_name})")
        
        # 从API获取数据
        df = fetch_specific_days(required_dates, interval=60000)  # 1分钟间隔
        
        if df.empty:
            logger.error("❌ 无法从API获取所需的历史数据")
            return False
        
        logger.info(f"✅ 成功获取 {len(df)} 条历史数据记录")
        
        # 设置时间戳为索引
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 保存原始数据
        self.raw_data = df.copy()
        
        logger.info(f"原始数据: {len(self.raw_data)} 行")
        logger.info(f"数据时间范围: {self.raw_data.index.min()} 至 {self.raw_data.index.max()}")
        
        # 保存预测信息
        self.prediction_info = {
            'prediction_date': prediction_date,
            'prediction_weekday_name': prediction_weekday_name,
            'three_weeks_ago_date': three_weeks_ago_date,
            'two_weeks_ago_date': two_weeks_ago_date,
            'last_week_date': last_week_date
        }
        
        return True
        
    def process_data(self, metric_type='flow'):
        """
        处理数据，构建训练数据
        
        Args:
            metric_type: 'flow' 或 'pressure'
        """
        logger.info(f"正在处理{metric_type}数据...")
        
        # 选择对应的数据列
        if metric_type == 'flow':
            target_column = '总流量'
        else:  # pressure
            target_column = '压力'
        
        # 数据平滑（可选）
        logger.info("正在平滑数据...")
        clean_data = self.raw_data[target_column].dropna()
        smoothed_values = gaussian_filter1d(clean_data.values, sigma=3)  # 减小平滑程度
        smoothed_data = pd.Series(smoothed_values, index=clean_data.index)
        
        # 保存平滑后的数据
        if metric_type == 'flow':
            self.smoothed_data_flow = smoothed_data
        else:
            self.smoothed_data_pressure = smoothed_data
        
        # 按日期分组数据
        df = pd.DataFrame({
            'value': smoothed_data,
            'date': smoothed_data.index.date,
            'hour': smoothed_data.index.hour,
            'minute': smoothed_data.index.minute
        })
        
        # 获取三个特定日期的数据
        three_weeks_ago_date = self.prediction_info['three_weeks_ago_date']
        two_weeks_ago_date = self.prediction_info['two_weeks_ago_date']
        last_week_date = self.prediction_info['last_week_date']
        
        # 分别获取三天的数据
        three_weeks_data = df[df['date'] == three_weeks_ago_date].copy()
        two_weeks_data = df[df['date'] == two_weeks_ago_date].copy()
        last_week_data = df[df['date'] == last_week_date].copy()
        
        logger.info(f"上上上周{metric_type}数据 ({three_weeks_ago_date}): {len(three_weeks_data)} 个数据点")
        logger.info(f"上上周{metric_type}数据 ({two_weeks_ago_date}): {len(two_weeks_data)} 个数据点")
        logger.info(f"上周{metric_type}数据 ({last_week_date}): {len(last_week_data)} 个数据点")
        
        # 检查数据完整性
        if len(three_weeks_data) < 1000 or len(two_weeks_data) < 1000 or len(last_week_data) < 1000:
            logger.error(f"❌ {metric_type}历史数据不够完整，无法进行可靠预测")
            logger.error(f"   需要每天至少1000个数据点")
            logger.error(f"   实际: 3周前={len(three_weeks_data)}, 2周前={len(two_weeks_data)}, 1周前={len(last_week_data)}")
            return False
        
        # 按时间排序
        three_weeks_data = three_weeks_data.sort_values(['hour', 'minute'])
        two_weeks_data = two_weeks_data.sort_values(['hour', 'minute'])
        last_week_data = last_week_data.sort_values(['hour', 'minute'])
        
        # 构建训练数据：按时间顺序放入三周的数据
        training_data_list = []
        training_data_list.extend(three_weeks_data['value'].values)  # 最早的数据
        training_data_list.extend(two_weeks_data['value'].values)    # 中间的数据
        training_data_list.extend(last_week_data['value'].values)    # 最近的数据
        
        # 创建连续的时间索引用于训练
        start_time = pd.Timestamp('2024-01-01 00:00:00')  # 虚拟起始时间
        time_index = pd.date_range(start=start_time, periods=len(training_data_list), freq='1min')
        
        training_data = pd.Series(training_data_list, index=time_index)
        
        # 保存训练数据
        if metric_type == 'flow':
            self.training_data_flow = training_data
        else:
            self.training_data_pressure = training_data
        
        logger.info(f"✅ 构建{metric_type}训练数据完成: {len(training_data)} 个数据点")
        logger.info(f"   三周数据总长度: {len(three_weeks_data)} + {len(two_weeks_data)} + {len(last_week_data)} = {len(training_data_list)}")
        logger.info(f"训练数据范围: {training_data.min():.2f} ~ {training_data.max():.2f}")
        logger.info(f"训练数据均值: {training_data.mean():.2f}")
        
        return True
        
    def predict(self, metric_type='flow', prediction_days=1):
        """
        使用 Holt-Winters 方法进行预测
        
        Args:
            metric_type: 'flow' 或 'pressure'
            prediction_days: 预测天数（默认1天）
        """
        logger.info(f"正在使用 Holt-Winters 方法进行{metric_type}预测...")
        
        try:
            # 选择对应的训练数据
            if metric_type == 'flow':
                training_data = self.training_data_flow
            else:
                training_data = self.training_data_pressure
            
            # 数据质量检查
            if training_data.isna().sum() > 0:
                logger.info(f"发现 {training_data.isna().sum()} 个缺失值，进行插值")
                training_data = training_data.interpolate(method='linear')
            
            # 设置季节性周期为1440分钟(1天)
            seasonal_periods = 1440
            
            if len(training_data) < seasonal_periods * 2:
                seasonal_periods = min(len(training_data) // 4, 720)  # 调整为4分之一，最大12小时周期
                seasonal_periods = max(seasonal_periods, 60)  # 最小1小时周期
                logger.info(f"数据较少，调整季节性周期为: {seasonal_periods} 分钟")
            else:
                logger.info(f"使用季节性周期: {seasonal_periods} 分钟 (1天)")
            
            # 创建 Holt-Winters 模型
            logger.info("正在创建 Holt-Winters 模型...")
            
            model = ExponentialSmoothing(
                training_data,
                trend=None,  # 无趋势，更稳定
                seasonal='add',  # 加法季节性
                seasonal_periods=seasonal_periods,
                damped_trend=False,
                initialization_method='estimated',
                use_boxcox=False
            )
            
            # 拟合模型
            fitted_model = model.fit(optimized=True, remove_bias=False)
            
            # 进行预测
            prediction_steps = prediction_days * 1440  # 转换为分钟数
            forecast = fitted_model.forecast(steps=prediction_steps)
            
            logger.info(f"原始预测值范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
            logger.info(f"原始预测值均值: {forecast.mean():.2f}")
            
            # 智能修正预测值
            data_mean = training_data.mean()
            data_std = training_data.std()
            data_min = training_data.min()
            data_max = training_data.max()
            
            # 1. 偏差校正
            prediction_bias = forecast.mean() - data_mean
            if abs(prediction_bias) > data_std * 0.12:  # 进一步降低阈值，因为有更多训练数据
                logger.info(f"检测到预测偏差 {prediction_bias:.2f}，进行校正")
                correction_factor = data_mean / forecast.mean()
                if 0.85 <= correction_factor <= 1.15:  # 缩小校正范围
                    forecast = forecast * correction_factor
                    logger.info(f"应用乘法校正因子: {correction_factor:.3f}")
                else:
                    forecast = forecast - prediction_bias
                    logger.info(f"应用加法校正: {-prediction_bias:.2f}")
            
            # 2. 处理负值
            negative_mask = forecast < 0
            if negative_mask.any():
                logger.info(f"发现 {negative_mask.sum()} 个负值，进行修正")
                replacement_value = max(data_min * 0.95, data_mean * 0.03)
                forecast[negative_mask] = replacement_value
            
            # 3. 处理极值
            upper_limit = data_max * 1.15  # 进一步降低上限
            lower_limit = data_min * 0.85  # 进一步提高下限
            
            extreme_high_mask = forecast > upper_limit
            extreme_low_mask = forecast < lower_limit
            
            if extreme_high_mask.any():
                logger.info(f"发现 {extreme_high_mask.sum()} 个过高值，限制在 {upper_limit:.2f}")
                forecast[extreme_high_mask] = upper_limit
            
            if extreme_low_mask.any():
                logger.info(f"发现 {extreme_low_mask.sum()} 个过低值，限制在 {lower_limit:.2f}")
                forecast[extreme_low_mask] = lower_limit
            
            # 4. 轻度平滑处理
            forecast = pd.Series(forecast).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            
            logger.info(f"最终预测值范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
            logger.info(f"最终预测值均值: {forecast.mean():.2f}")
            logger.info(f"与训练数据均值偏差: {forecast.mean() - data_mean:.2f}")
            
            # 创建预测时间索引
            prediction_date = self.prediction_info['prediction_date']
            prediction_start_time = pd.Timestamp(f"{prediction_date} 00:00:00")
            prediction_index = pd.date_range(
                start=prediction_start_time,
                periods=len(forecast),
                freq='1min'
            )
            
            prediction_series = pd.Series(forecast, index=prediction_index)
            
            # 保存预测信息到结果中
            prediction_series.prediction_info = self.prediction_info.copy()
            prediction_series.prediction_info['metric_type'] = metric_type
            
            logger.info(f"✅ {metric_type}预测完成: {len(prediction_series)} 个数据点")
            return prediction_series
            
        except Exception as e:
            logger.error(f"❌ {metric_type}预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_prediction_results(self, prediction_series, save_path='test_prediction_plot.png'):
        """绘制预测结果图"""
        logger.info("正在绘制预测结果...")
        
        if not hasattr(prediction_series, 'prediction_info'):
            logger.error("❌ 缺少预测信息，无法绘制详细图表")
            return False
        
        info = prediction_series.prediction_info
        prediction_date = info['prediction_date']
        prediction_weekday_name = info['prediction_weekday_name']
        three_weeks_ago_date = info['three_weeks_ago_date']
        two_weeks_ago_date = info['two_weeks_ago_date']
        last_week_date = info['last_week_date']
        metric_type = info['metric_type']
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # 获取对应的平滑数据
        if metric_type == 'flow':
            smoothed_data = self.smoothed_data_flow
            ylabel = '瞬时流量'
        else:
            smoothed_data = self.smoothed_data_pressure  
            ylabel = '总压力'
        
        # 从原始数据中获取三天的历史数据
        df = pd.DataFrame({
            'value': smoothed_data,
            'date': smoothed_data.index.date,
            'hour': smoothed_data.index.hour,
            'minute': smoothed_data.index.minute
        })
        
        three_weeks_data = df[df['date'] == three_weeks_ago_date].copy()
        two_weeks_data = df[df['date'] == two_weeks_ago_date].copy()
        last_week_data = df[df['date'] == last_week_date].copy()
        
        if len(three_weeks_data) > 0 and len(two_weeks_data) > 0 and len(last_week_data) > 0:
            # 创建x轴标签（24小时）
            hours = list(range(24))
            
            # 计算每小时平均值
            def get_hourly_averages(data):
                hourly_values = []
                data_sorted = data.sort_values(['hour', 'minute'])
                for hour in range(24):
                    hour_data = data_sorted[data_sorted['hour'] == hour]
                    if len(hour_data) > 0:
                        hourly_values.append(hour_data['value'].mean())
                    else:
                        hourly_values.append(None)
                return hourly_values
            
            three_weeks_values = get_hourly_averages(three_weeks_data)
            two_weeks_values = get_hourly_averages(two_weeks_data)
            last_week_values = get_hourly_averages(last_week_data)
            
            # 预测数据按小时平均
            prediction_df = pd.DataFrame({
                'value': prediction_series.values,
                'hour': prediction_series.index.hour
            })
            
            prediction_values = []
            for hour in range(24):
                hour_data = prediction_df[prediction_df['hour'] == hour]
                if len(hour_data) > 0:
                    prediction_values.append(hour_data['value'].mean())
                else:
                    prediction_values.append(None)
            
            # 绘制四条线
            ax.plot(hours, three_weeks_values, 
                   color='purple', linewidth=2.5, marker='d', markersize=4,
                   label=f'3周前数据: {three_weeks_ago_date} ({prediction_weekday_name})', alpha=0.8)
            
            ax.plot(hours, two_weeks_values, 
                   color='blue', linewidth=2.5, marker='o', markersize=4,
                   label=f'2周前数据: {two_weeks_ago_date} ({prediction_weekday_name})', alpha=0.8)
            
            ax.plot(hours, last_week_values, 
                   color='green', linewidth=2.5, marker='s', markersize=4,
                   label=f'1周前数据: {last_week_date} ({prediction_weekday_name})', alpha=0.8)
            
            ax.plot(hours, prediction_values, 
                   color='red', linewidth=3, marker='^', markersize=5,
                   label=f'预测: {prediction_date} ({prediction_weekday_name})', linestyle='--')
            
            # 设置图表
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 24, 2)])
            ax.set_title(f'{ylabel}预测对比 - {prediction_weekday_name} (API数据 - 3周训练)', 
                       fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            
            # 添加说明文本
            info_text = f"训练数据:\n• 3周前: {three_weeks_ago_date}\n• 2周前: {two_weeks_ago_date}\n• 1周前: {last_week_date}\n• 预测: {prediction_date}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 确保保存路径存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 保存图片
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图已保存到: {os.path.abspath(save_path)}")
            plt.close()
            return True
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            plt.close()
            return False

# 全局预测器实例
predictor = APIGasPredictor()

def perform_prediction(target_date, metric_type='flow'):
    """
    执行预测的核心函数
    
    Args:
        target_date: 预测日期
        metric_type: 'flow' 或 'pressure'
    
    Returns:
        预测结果或None
    """
    try:
        # 加载数据
        if not predictor.load_specific_dates_from_api(target_date):
            return None
        
        # 处理数据
        if not predictor.process_data(metric_type):
            return None
        
        # 进行预测
        result = predictor.predict(metric_type)
        return result
        
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        return None

@app.get('/predict', response_model=PredictionResponse)
async def predict_flow(date: str = Query(..., description="预测日期，格式为YYYY-MM-DD")):
    """流量预测接口"""
    try:
        # 解析日期
        try:
            target_date = pd.to_datetime(date).date()
        except:
            raise HTTPException(
                status_code=400,
                detail={"error": "日期格式错误", "message": "请使用YYYY-MM-DD格式"}
            )
        
        logger.info(f"收到流量预测请求: {target_date}")
        
        # 执行预测
        result = perform_prediction(target_date, 'flow')
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "预测失败", "message": "无法获取足够的历史数据或预测过程出错"}
            )
        
        # 返回结果
        predictions = []
        for timestamp, forecast_value in result.items():
            predictions.append(PredictionPoint(
                timestamp=timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                forecast=float(forecast_value)
            ))
        
        return PredictionResponse(
            success=True,
            prediction_date=str(target_date),
            metric='瞬时流量',
            data_points=len(predictions),
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流量预测接口错误: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "内部服务器错误", "message": str(e)}
        )

@app.get('/predict_pressure', response_model=PredictionResponse)
async def predict_pressure(date: str = Query(..., description="预测日期，格式为YYYY-MM-DD")):
    """压力预测接口"""
    try:
        # 解析日期
        try:
            target_date = pd.to_datetime(date).date()
        except:
            raise HTTPException(
                status_code=400,
                detail={"error": "日期格式错误", "message": "请使用YYYY-MM-DD格式"}
            )
        
        logger.info(f"收到压力预测请求: {target_date}")
        
        # 执行预测
        result = perform_prediction(target_date, 'pressure')
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "预测失败", "message": "无法获取足够的历史数据或预测过程出错"}
            )
        
        # 返回结果
        predictions = []
        for timestamp, forecast_value in result.items():
            predictions.append(PredictionPoint(
                timestamp=timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                forecast=float(forecast_value)
            ))
        
        return PredictionResponse(
            success=True,
            prediction_date=str(target_date),
            metric='总压力',
            data_points=len(predictions),
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"压力预测接口错误: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "内部服务器错误", "message": str(e)}
        )

@app.get('/test', response_model=TestResponse)
async def test_prediction(
    date: str = Query(..., description="预测日期，格式为YYYY-MM-DD"),
    metric: str = Query('flow', description="预测指标：flow(流量) 或 pressure(压力)")
):
    """测试预测接口 - 生成预测结果并返回可视化图表"""
    try:
        # 验证metric参数
        if metric not in ['flow', 'pressure']:
            raise HTTPException(
                status_code=400,
                detail={"error": "参数错误", "message": "metric参数必须是 'flow' 或 'pressure'"}
            )
        
        # 解析日期
        try:
            target_date = pd.to_datetime(date).date()
        except:
            raise HTTPException(
                status_code=400,
                detail={"error": "日期格式错误", "message": "请使用YYYY-MM-DD格式"}
            )
        
        logger.info(f"收到测试预测请求: {target_date}, 指标: {metric}")
        
        # 执行预测
        result = perform_prediction(target_date, metric)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "预测失败", "message": "无法获取足够的历史数据或预测过程出错"}
            )
        
        # 生成图表
        date_str = str(target_date).replace('-', '')
        plot_filename = f'test_{metric}_prediction_{date_str}.png'
        plot_path = os.path.join('static', plot_filename)
        
        # 确保static目录存在
        os.makedirs('static', exist_ok=True)
        
        success = predictor.plot_prediction_results(result, plot_path)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"error": "图表生成失败", "message": "无法生成预测可视化图表"}
            )
        
        metric_name = '瞬时流量' if metric == 'flow' else '总压力'
        
        return TestResponse(
            success=True,
            prediction_date=str(target_date),
            metric=metric_name,
            plot_file=plot_filename,
            message=f"预测完成，共生成{len(result)}个数据点。可通过 /static/{plot_filename} 查看预测图表。"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"测试预测接口错误: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "内部服务器错误", "message": str(e)}
        )

@app.get('/static/{filename}')
async def get_static_file(filename: str):
    """静态文件服务 - 用于访问生成的图片"""
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="文件未找到")

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status='healthy',
        service='天然气预测API',
        version='1.0.0'
    )

@app.get('/', response_model=ServiceInfo)
async def index():
    """根路径信息"""
    return ServiceInfo(
        service='天然气预测API服务',
        version='1.0.0',
        endpoints={
            'flow_prediction': '/predict?date=YYYY-MM-DD',
            'pressure_prediction': '/predict_pressure?date=YYYY-MM-DD',
            'test_prediction': '/test?date=YYYY-MM-DD&metric=flow/pressure',
            'static_files': '/static/{filename}',
            'health_check': '/health'
        },
        description='基于历史3周同期数据的天然气流量和压力预测服务'
    )

if __name__ == '__main__':
    print("🚀 启动天然气预测API服务器 (FastAPI)")
    print("=" * 50)
    print("可用接口:")
    print("  流量预测: http://127.0.0.1:58888/predict?date=2025-07-07")
    print("  压力预测: http://127.0.0.1:58888/predict_pressure?date=2025-07-07")
    print("  测试预测: http://127.0.0.1:58888/test?date=2025-07-07&metric=flow")
    print("  健康检查: http://127.0.0.1:58888/health")
    print("  API文档: http://127.0.0.1:58888/docs")
    print("=" * 50)
    
    uvicorn.run(app, host='127.0.0.1', port=58888) 
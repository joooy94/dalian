#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于API的天然气生产数据预测工具
直接使用API获取数据，只获取预测所需的历史同周期数据
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os
import sys
import warnings
from datetime import datetime, timedelta
import logging

# 导入API数据获取模块
from stl_decomposition.api_data_fetcher import fetch_specific_days

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Gas_Prediction")

class APIGasPredictor:
    def __init__(self):
        """初始化API预测器"""
        self.raw_data = None
        self.smoothed_data = None
        self.training_data = None
        
    def load_specific_dates_from_api(self, prediction_date):
        """
        从API加载预测所需的特定历史日期数据
        
        Args:
            prediction_date: 预测日期（字符串或datetime对象）
        """
        print("正在从API加载预测所需的历史数据...")
        
        # 确定预测日期
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date).date()
        else:
            prediction_date = pd.to_datetime(prediction_date).date()
        
        # 获取预测日期的星期几
        prediction_weekday = pd.to_datetime(prediction_date).weekday()
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        prediction_weekday_name = weekday_names[prediction_weekday]
        
        print(f"预测日期: {prediction_date} ({prediction_weekday_name})")
        
        # 计算需要的历史数据日期（上周、上上周、上上上周的同一天）
        last_week_date = prediction_date - timedelta(days=7)  # 上周同一天
        two_weeks_ago_date = prediction_date - timedelta(days=14)  # 上上周同一天
        three_weeks_ago_date = prediction_date - timedelta(days=21)  # 上上上周同一天
        
        required_dates = [
            three_weeks_ago_date.strftime("%Y-%m-%d"),
            two_weeks_ago_date.strftime("%Y-%m-%d"),
            last_week_date.strftime("%Y-%m-%d")
        ]
        
        print(f"需要获取的历史数据日期:")
        print(f"  上上上周同一天: {required_dates[0]} ({prediction_weekday_name})")
        print(f"  上上周同一天: {required_dates[1]} ({prediction_weekday_name})")
        print(f"  上周同一天: {required_dates[2]} ({prediction_weekday_name})")
        
        # 从API获取数据
        df = fetch_specific_days(required_dates, interval=600000)  # 1分钟间隔
        
        if df.empty:
            print("❌ 无法从API获取所需的历史数据")
            return False
        
        print(f"✅ 成功获取 {len(df)} 条历史数据记录")
        
        # 重命名列
        df.rename(columns={'总流量': '总系统瞬时（计算）'}, inplace=True)
        
        # 设置时间戳为索引
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 保存原始数据
        self.raw_data = df['总系统瞬时（计算）'].copy()
        
        print(f"原始数据: {len(self.raw_data)} 行")
        print(f"数据时间范围: {self.raw_data.index.min()} 至 {self.raw_data.index.max()}")
        
        # 保存预测信息
        self.prediction_info = {
            'prediction_date': prediction_date,
            'prediction_weekday_name': prediction_weekday_name,
            'three_weeks_ago_date': three_weeks_ago_date,
            'two_weeks_ago_date': two_weeks_ago_date,
            'last_week_date': last_week_date
        }
        
        return True
        
    def process_data(self):
        """处理数据，构建训练数据"""
        print("正在处理数据...")
        
        # 数据平滑（可选）
        print("正在平滑数据...")
        clean_data = self.raw_data.dropna()
        smoothed_values = gaussian_filter1d(clean_data.values, sigma=3)  # 减小平滑程度
        self.smoothed_data = pd.Series(smoothed_values, index=clean_data.index)
        
        # 按日期分组数据
        df = pd.DataFrame({
            'value': self.smoothed_data,
            'date': self.smoothed_data.index.date,
            'hour': self.smoothed_data.index.hour,
            'minute': self.smoothed_data.index.minute
        })
        
        # 获取三个特定日期的数据
        three_weeks_ago_date = self.prediction_info['three_weeks_ago_date']
        two_weeks_ago_date = self.prediction_info['two_weeks_ago_date']
        last_week_date = self.prediction_info['last_week_date']
        
        # 分别获取三天的数据
        three_weeks_data = df[df['date'] == three_weeks_ago_date].copy()
        two_weeks_data = df[df['date'] == two_weeks_ago_date].copy()
        last_week_data = df[df['date'] == last_week_date].copy()
        
        print(f"上上上周数据 ({three_weeks_ago_date}): {len(three_weeks_data)} 个数据点")
        print(f"上上周数据 ({two_weeks_ago_date}): {len(two_weeks_data)} 个数据点")
        print(f"上周数据 ({last_week_date}): {len(last_week_data)} 个数据点")
        
        # 检查数据完整性
        if len(three_weeks_data) < 1000 or len(two_weeks_data) < 1000 or len(last_week_data) < 1000:
            print("❌ 历史数据不够完整，无法进行可靠预测")
            print(f"   需要每天至少1000个数据点")
            print(f"   实际: 3周前={len(three_weeks_data)}, 2周前={len(two_weeks_data)}, 1周前={len(last_week_data)}")
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
        
        self.training_data = pd.Series(training_data_list, index=time_index)
        
        print(f"✅ 构建训练数据完成: {len(self.training_data)} 个数据点")
        print(f"   三周数据总长度: {len(three_weeks_data)} + {len(two_weeks_data)} + {len(last_week_data)} = {len(training_data_list)}")
        print(f"训练数据范围: {self.training_data.min():.2f} ~ {self.training_data.max():.2f}")
        print(f"训练数据均值: {self.training_data.mean():.2f}")
        
        return True
        
    def predict(self, prediction_days=1):
        """
        使用 Holt-Winters 方法进行预测
        
        Args:
            prediction_days: 预测天数（默认1天）
        """
        print("正在使用 Holt-Winters 方法进行预测...")
        
        try:
            # 数据质量检查
            if self.training_data.isna().sum() > 0:
                print(f"发现 {self.training_data.isna().sum()} 个缺失值，进行插值")
                self.training_data = self.training_data.interpolate(method='linear')
            
            # 设置季节性周期为1440分钟(1天)
            seasonal_periods = 1440
            
            if len(self.training_data) < seasonal_periods * 2:
                seasonal_periods = min(len(self.training_data) // 4, 720)  # 调整为4分之一，最大12小时周期
                seasonal_periods = max(seasonal_periods, 60)  # 最小1小时周期
                print(f"数据较少，调整季节性周期为: {seasonal_periods} 分钟")
            else:
                print(f"使用季节性周期: {seasonal_periods} 分钟 (1天)")
            
            # 创建 Holt-Winters 模型
            print("正在创建 Holt-Winters 模型...")
            
            model = ExponentialSmoothing(
                self.training_data,
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
            
            print(f"原始预测值范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
            print(f"原始预测值均值: {forecast.mean():.2f}")
            
            # 智能修正预测值
            data_mean = self.training_data.mean()
            data_std = self.training_data.std()
            data_min = self.training_data.min()
            data_max = self.training_data.max()
            
            # 1. 偏差校正
            prediction_bias = forecast.mean() - data_mean
            if abs(prediction_bias) > data_std * 0.12:  # 进一步降低阈值，因为有更多训练数据
                print(f"检测到预测偏差 {prediction_bias:.2f}，进行校正")
                correction_factor = data_mean / forecast.mean()
                if 0.85 <= correction_factor <= 1.15:  # 缩小校正范围
                    forecast = forecast * correction_factor
                    print(f"应用乘法校正因子: {correction_factor:.3f}")
                else:
                    forecast = forecast - prediction_bias
                    print(f"应用加法校正: {-prediction_bias:.2f}")
            
            # 2. 处理负值
            negative_mask = forecast < 0
            if negative_mask.any():
                print(f"发现 {negative_mask.sum()} 个负值，进行修正")
                replacement_value = max(data_min * 0.95, data_mean * 0.03)
                forecast[negative_mask] = replacement_value
            
            # 3. 处理极值
            upper_limit = data_max * 1.15  # 进一步降低上限
            lower_limit = data_min * 0.85  # 进一步提高下限
            
            extreme_high_mask = forecast > upper_limit
            extreme_low_mask = forecast < lower_limit
            
            if extreme_high_mask.any():
                print(f"发现 {extreme_high_mask.sum()} 个过高值，限制在 {upper_limit:.2f}")
                forecast[extreme_high_mask] = upper_limit
            
            if extreme_low_mask.any():
                print(f"发现 {extreme_low_mask.sum()} 个过低值，限制在 {lower_limit:.2f}")
                forecast[extreme_low_mask] = lower_limit
            
            # 4. 轻度平滑处理
            forecast = pd.Series(forecast).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            
            print(f"最终预测值范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
            print(f"最终预测值均值: {forecast.mean():.2f}")
            print(f"与训练数据均值偏差: {forecast.mean() - data_mean:.2f}")
            
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
            prediction_series.prediction_info = self.prediction_info
            
            print(f"✅ 预测完成: {len(prediction_series)} 个数据点")
            return prediction_series
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_prediction_results(self, prediction_series, save_path='api_prediction_plot.png'):
        """绘制预测结果图"""
        print("正在绘制预测结果...")
        
        if not hasattr(prediction_series, 'prediction_info'):
            print("❌ 缺少预测信息，无法绘制详细图表")
            return
        
        info = prediction_series.prediction_info
        prediction_date = info['prediction_date']
        prediction_weekday_name = info['prediction_weekday_name']
        three_weeks_ago_date = info['three_weeks_ago_date']
        two_weeks_ago_date = info['two_weeks_ago_date']
        last_week_date = info['last_week_date']
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # 从原始数据中获取三天的历史数据
        df = pd.DataFrame({
            'value': self.smoothed_data,
            'date': self.smoothed_data.index.date,
            'hour': self.smoothed_data.index.hour,
            'minute': self.smoothed_data.index.minute
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
            ax.set_title(f'天然气生产预测对比 - {prediction_weekday_name} (API数据 - 3周训练)', 
                       fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('时间 (小时)', fontsize=12)
            ax.set_ylabel('总系统瞬时（计算）', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
            
            # 添加说明文本
            info_text = f"训练数据:\n• 3周前: {three_weeks_ago_date}\n• 2周前: {two_weeks_ago_date}\n• 1周前: {last_week_date}\n• 预测: {prediction_date}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到: {os.path.abspath(save_path)}")
        plt.close()


def main():
    """主函数"""
    print("🔮 API天然气预测工具 (3周训练版本)")
    print("=" * 50)
    
    # 解析命令行参数
    custom_prediction_date = None
    if len(sys.argv) > 1:
        try:
            date_str = sys.argv[1]
            custom_prediction_date = pd.to_datetime(date_str).date()
            print(f"📅 指定预测日期: {custom_prediction_date}")
        except Exception as e:
            print(f"❌ 日期格式错误: {e}")
            print("请使用格式: YYYY-MM-DD")
            print("例如: python api_gas_prediction.py 2025-07-07")
            sys.exit(1)
    else:
        print("❌ 请指定预测日期")
        print("用法: python api_gas_prediction.py 2025-07-07")
        sys.exit(1)
    
    # 创建预测器
    predictor = APIGasPredictor()
    
    try:
        # 从API加载所需的历史数据
        print(f"\n📊 从API加载预测所需的历史数据...")
        if not predictor.load_specific_dates_from_api(custom_prediction_date):
            print("❌ 无法获取所需的历史数据")
            sys.exit(1)
        
        # 处理数据
        print("\n📊 处理数据...")
        if not predictor.process_data():
            print("❌ 数据处理失败")
            sys.exit(1)
        
        # 进行预测
        print(f"\n🔮 开始预测...")
        result = predictor.predict()
        
        if result is not None:
            print(f"\n✅ 预测成功!")
            print(f"预测值范围: {result.min():.2f} ~ {result.max():.2f}")
            print(f"预测值均值: {result.mean():.2f}")
            print(f"预测值中位数: {result.median():.2f}")
            
            # 生成文件名
            date_str = str(custom_prediction_date).replace('-', '')
            csv_filename = f'api_prediction_3weeks_{date_str}.csv'
            plot_filename = f'api_prediction_3weeks_{date_str}.png'
            
            # 显示详细信息
            info = result.prediction_info
            print(f"\n📈 预测详情:")
            print(f"   预测日期: {info['prediction_date']} ({info['prediction_weekday_name']})")
            print(f"   使用的训练数据:")
            print(f"     - 3周前同一天: {info['three_weeks_ago_date']}")
            print(f"     - 2周前同一天: {info['two_weeks_ago_date']}")
            print(f"     - 1周前同一天: {info['last_week_date']}")
            print(f"   预测时间段: 全天24小时 (1440个数据点)")
            
            # 保存结果
            result.to_csv(csv_filename, header=['predicted_value'])
            print(f"\n💾 结果已保存:")
            print(f"   CSV文件: {csv_filename}")
            
            # 绘制图片
            predictor.plot_prediction_results(result, plot_filename)
            print(f"   图片文件: {plot_filename}")
            
            # 显示关键时段预测值
            print(f"\n⏰ 关键时段预测值:")
            for hour in [6, 12, 18, 23]:
                hour_data = result[result.index.hour == hour]
                if len(hour_data) > 0:
                    avg_value = hour_data.mean()
                    print(f"   {hour:02d}:00 时段均值: {avg_value:.2f}")
            
        else:
            print("\n❌ 预测失败!")
            
            last_week = custom_prediction_date - timedelta(days=7)
            two_weeks_ago = custom_prediction_date - timedelta(days=14)
            three_weeks_ago = custom_prediction_date - timedelta(days=21)
            
            print("可能原因:")
            print(f"  1. API中没有足够的历史数据")
            print(f"     需要: {three_weeks_ago}, {two_weeks_ago}, {last_week}")
            print(f"  2. 这些日期的数据不完整（少于1000个数据点/天）")
            print(f"  3. API连接问题")
            
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
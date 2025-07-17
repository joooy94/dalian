#!/opt/anaconda3/envs/water/bin/python
# -*- coding: utf-8 -*-
"""
天然气生产数据预测工具 - 核心版本
基于历史同周期匹配法进行预测
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GasPredictor:
    def __init__(self, excel_file='test.xlsx'):
        """初始化预测器"""
        self.excel_file = excel_file
        self.raw_data = None
        self.smoothed_data = None
        self.resampled_data = None
        self.weekday_patterns = None
        
    def load_and_process_data(self):
        """加载和处理数据"""
        print("正在加载数据...")
        
        # 读取数据
        df = pd.read_excel(self.excel_file, sheet_name='Sheet2')
        df['TagTime'] = pd.to_datetime(df['TagTime'])
        df.set_index('TagTime', inplace=True)
        df.sort_index(inplace=True)
        
        # 提取目标列
        self.raw_data = df['总系统瞬时（计算）'].copy()
        print(f"原始数据: {len(self.raw_data)} 行")
        
        # 数据平滑
        print("正在平滑数据...")
        clean_data = self.raw_data.dropna()
        smoothed_values = gaussian_filter1d(clean_data.values, sigma=5)
        self.smoothed_data = pd.Series(smoothed_values, index=clean_data.index)
        
        # 重采样到1分钟
        print("正在重采样数据...")
        self.resampled_data = self.smoothed_data.resample('1min').mean()
        self.resampled_data = self.resampled_data.interpolate(method='time')
        print(f"重采样后: {len(self.resampled_data)} 行")
        
        # 提取工作日模式
        print("正在提取工作日模式...")
        df = pd.DataFrame({
            'value': self.resampled_data,
            'weekday': self.resampled_data.index.dayofweek,
            'hour': self.resampled_data.index.hour,
            'minute': self.resampled_data.index.minute,
            'date': self.resampled_data.index.date
        })
        
        weekday_patterns = {}
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        for weekday in range(7):
            weekday_name = weekday_names[weekday]
            weekday_data = df[df['weekday'] == weekday].copy()
            weekday_patterns[weekday_name] = weekday_data
            print(f"{weekday_name}: {len(weekday_data)} 个数据点")
        
        self.weekday_patterns = weekday_patterns
        
    def predict(self, prediction_days=1, custom_prediction_date=None):
        """
        使用 Holt-Winters 方法进行预测
        基于上周和上上周同一天的数据（48小时）
        
        Args:
            prediction_days: 预测天数（默认1天）
            custom_prediction_date: 自定义预测日期，格式为'YYYY-MM-DD'或datetime对象
        """
        print("正在使用 Holt-Winters 方法进行预测...")
        
        try:
            # 确定预测日期
            if custom_prediction_date is not None:
                if isinstance(custom_prediction_date, str):
                    prediction_date = pd.to_datetime(custom_prediction_date).date()
                else:
                    prediction_date = pd.to_datetime(custom_prediction_date).date()
                print(f"使用自定义预测日期: {prediction_date}")
            else:
                # 默认预测明天
                last_date = self.resampled_data.index.max().date()
                prediction_date = last_date + pd.Timedelta(days=1)
                print(f"使用默认预测日期: {prediction_date} (最新数据后一天)")
            
            # 获取预测日期的星期几
            prediction_weekday = pd.to_datetime(prediction_date).weekday()
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            prediction_weekday_name = weekday_names[prediction_weekday]
            
            print(f"预测日期: {prediction_date} ({prediction_weekday_name})")
            
            # 计算需要的历史数据日期
            last_week_date = prediction_date - pd.Timedelta(days=7)  # 上周同一天
            two_weeks_ago_date = prediction_date - pd.Timedelta(days=14)  # 上上周同一天
            
            print(f"需要的历史数据:")
            print(f"  上周同一天: {last_week_date} ({prediction_weekday_name})")
            print(f"  上上周同一天: {two_weeks_ago_date} ({prediction_weekday_name})")
            
            # 获取该工作日的历史数据
            weekday_data = self.weekday_patterns[prediction_weekday_name]
            
            if len(weekday_data) == 0:
                print(f"❌ 没有找到{prediction_weekday_name}的历史数据")
                return None
            
            # 按日期分组，查找所需的具体日期
            daily_data = weekday_data.groupby('date')
            available_dates = list(daily_data.groups.keys())
            
            print(f"找到{prediction_weekday_name}的所有历史日期: {sorted(available_dates)}")
            
            # 检查所需的两个日期是否存在
            required_dates = [last_week_date, two_weeks_ago_date]
            available_valid_dates = []
            
            for required_date in required_dates:
                if required_date in available_dates:
                    day_data = daily_data.get_group(required_date)
                    if len(day_data) >= 200:  # 数据完整性检查
                        available_valid_dates.append(required_date)
                        print(f"✅ {required_date}: {len(day_data)} 个数据点 (完整)")
                    else:
                        print(f"❌ {required_date}: {len(day_data)} 个数据点 (数据不完整)")
                else:
                    print(f"❌ {required_date}: 没有找到数据")
            
            # 如果找不到所需的两个日期，尝试寻找替代方案
            if len(available_valid_dates) < 2:
                print(f"\n⚠️  无法找到足够的指定日期数据，尝试寻找替代方案...")
                
                # 找到所有完整的该工作日数据
                valid_alternative_dates = []
                for date in sorted(available_dates, reverse=True):  # 从最新开始
                    day_data = daily_data.get_group(date)
                    if len(day_data) >= 200:
                        valid_alternative_dates.append(date)
                        print(f"📅 可用日期: {date} ({len(day_data)} 个数据点)")
                        if len(valid_alternative_dates) >= 2:
                            break
                
                if len(valid_alternative_dates) < 2:
                    print(f"❌ 找不到足够的完整{prediction_weekday_name}数据，无法进行预测")
                    print(f"   需要至少2天完整数据，实际找到: {len(valid_alternative_dates)} 天")
                    return None
                else:
                    # 使用最近的两个完整日期作为替代
                    available_valid_dates = valid_alternative_dates[:2]
                    print(f"✅ 使用替代日期: {available_valid_dates}")
            
            # 构建训练数据
            if len(available_valid_dates) >= 2:
                # 按时间顺序排序（早的在前）
                available_valid_dates.sort()
                two_weeks_data_date = available_valid_dates[0]  # 较早的日期
                last_week_data_date = available_valid_dates[1]   # 较晚的日期
                
                print(f"\n🔧 构建训练数据:")
                print(f"   第一部分: {two_weeks_data_date}")
                print(f"   第二部分: {last_week_data_date}")
                
                # 获取两天的数据
                last_week_data = daily_data.get_group(last_week_data_date).sort_values(['hour', 'minute'])
                two_weeks_ago_data = daily_data.get_group(two_weeks_data_date).sort_values(['hour', 'minute'])
                
                # 将两天数据合并（早期数据在前，较近数据在后）
                training_data_list = []
                
                # 添加较早的数据
                training_data_list.extend(two_weeks_ago_data['value'].values)
                
                # 添加较近的数据
                training_data_list.extend(last_week_data['value'].values)
                
                # 创建连续的时间索引
                start_time = pd.Timestamp('2024-01-01 00:00:00')  # 虚拟起始时间
                time_index = pd.date_range(start=start_time, periods=len(training_data_list), freq='1min')
                
                training_data = pd.Series(training_data_list, index=time_index)
                
                print(f"✅ 合并训练数据长度: {len(training_data)} 个数据点")
                print(f"   第一部分数据: {len(two_weeks_ago_data)} 点")
                print(f"   第二部分数据: {len(last_week_data)} 点")
            else:
                print("❌ 无法构建训练数据")
                return None
            
            print(f"训练数据范围: {training_data.min():.2f} ~ {training_data.max():.2f}")
            print(f"训练数据均值: {training_data.mean():.2f}")
            
            # 数据质量检查
            if training_data.isna().sum() > 0:
                print(f"发现 {training_data.isna().sum()} 个缺失值，进行插值")
                training_data = training_data.interpolate(method='linear')
            
            # 检查数据方差
            if training_data.std() < 1e-6:
                print("警告: 训练数据方差过小，可能影响预测质量")
            
            # 设置季节性周期为1440分钟(1天)
            seasonal_periods = 1440
            
            # 调整季节性周期，确保合理
            if len(training_data) < seasonal_periods * 2:
                seasonal_periods = min(len(training_data) // 3, 360)  # 最大6小时周期
                seasonal_periods = max(seasonal_periods, 60)  # 最小1小时周期
                print(f"数据较少，调整季节性周期为: {seasonal_periods} 分钟")
            else:
                print(f"使用季节性周期: {seasonal_periods} 分钟")
            
            # 创建 Holt-Winters 模型 - 使用更保守的参数
            print("正在创建 Holt-Winters 模型...")
            
            # 尝试不同的模型配置
            model_configs = [
                # 配置1: 无趋势加法模型（最稳定）
                {'trend': None, 'seasonal': 'add', 'damped_trend': False},
                # # 配置2: 阻尼加法模型
                # {'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
                # # 配置3: 简单指数平滑（备选）
                # {'trend': None, 'seasonal': None, 'damped_trend': False},
                # # 配置4: 乘法季节性（最后尝试，容易偏大）
                # {'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
            ]
            
            best_model = None
            best_forecast = None
            best_bias = float('inf')
            
            for i, config in enumerate(model_configs):
                try:
                    print(f"尝试配置 {i+1}: {config}")
                    
                    model = ExponentialSmoothing(
                        training_data,
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=seasonal_periods if config['seasonal'] else None,
                        damped_trend=config['damped_trend'],
                        initialization_method='estimated',
                        use_boxcox=False
                    )
                    
                    # 拟合模型
                    fitted_model = model.fit(optimized=True, remove_bias=False)
                    
                    # 进行预测
                    prediction_steps = prediction_days * 1440  # 转换为分钟数
                    forecast = fitted_model.forecast(steps=prediction_steps)
                    
                    # 检查预测质量
                    negative_count = (forecast < 0).sum()
                    extreme_count = (forecast > training_data.max() * 2).sum()
                    
                    # 计算预测偏差（关键改进）
                    prediction_bias = abs(forecast.mean() - training_data.mean())
                    
                    print(f"  预测范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
                    print(f"  预测均值: {forecast.mean():.2f} (训练数据均值: {training_data.mean():.2f})")
                    print(f"  预测偏差: {prediction_bias:.2f}")
                    print(f"  负值数量: {negative_count}, 极值数量: {extreme_count}")
                    
                    # 选择偏差最小且质量合格的模型
                    if negative_count == 0 and extreme_count < len(forecast) * 0.1:  # 极值不超过10%
                        if prediction_bias < best_bias:
                            best_model = fitted_model
                            best_forecast = forecast.copy()
                            best_bias = prediction_bias
                            print(f"  → 选择此配置 (偏差最小)")
                        else:
                            print(f"  → 偏差较大，继续尝试")
                    else:
                        print(f"  → 质量不佳，跳过")
                    
                except Exception as e:
                    print(f"  配置 {i+1} 失败: {e}")
                    continue
            
            if best_model is None:
                print("所有模型配置都失败，无法进行预测")
                return None
            
            forecast = best_forecast
            print(f"\n✅ 选择了偏差最小的模型，预测偏差: {best_bias:.2f}")
            
            # 记录原始预测值
            original_min = forecast.min()
            original_max = forecast.max()
            original_mean = forecast.mean()
            print(f"原始预测值范围: {original_min:.2f} ~ {original_max:.2f}")
            print(f"原始预测值均值: {original_mean:.2f}")
            
            # 智能修正预测值
            data_mean = training_data.mean()
            data_std = training_data.std()
            data_min = training_data.min()
            data_max = training_data.max()
            
            # 1. 偏差校正 - 如果预测均值偏离训练数据均值太多，进行校正
            prediction_bias = forecast.mean() - data_mean
            if abs(prediction_bias) > data_std * 0.2:  # 如果偏差超过0.2个标准差
                print(f"检测到较大预测偏差 {prediction_bias:.2f}，进行校正")
                
                # 计算校正因子
                correction_factor = data_mean / forecast.mean()
                if 0.7 <= correction_factor <= 1.3:  # 只在合理范围内校正
                    forecast = forecast * correction_factor
                    print(f"应用乘法校正因子: {correction_factor:.3f}")
                else:
                    # 使用加法校正
                    forecast = forecast - prediction_bias
                    print(f"应用加法校正: {-prediction_bias:.2f}")
            
            # 2. 处理负值 - 使用更智能的方法
            negative_mask = forecast < 0
            if negative_mask.any():
                print(f"发现 {negative_mask.sum()} 个负值，进行修正")
                
                # 对负值使用历史数据的最小值或均值的某个比例
                replacement_value = max(data_min * 0.8, data_mean * 0.1)
                forecast[negative_mask] = replacement_value
                print(f"负值替换为: {replacement_value:.2f}")
            
            # 3. 处理极值 - 限制在合理范围内
            upper_limit = data_max * 1.3  # 降低上限（从1.5到1.3）
            lower_limit = data_min * 0.7   # 增加下限
            
            extreme_high_mask = forecast > upper_limit
            extreme_low_mask = forecast < lower_limit
            
            if extreme_high_mask.any():
                print(f"发现 {extreme_high_mask.sum()} 个过高值，限制在 {upper_limit:.2f}")
                forecast[extreme_high_mask] = upper_limit
            
            if extreme_low_mask.any():
                print(f"发现 {extreme_low_mask.sum()} 个过低值，限制在 {lower_limit:.2f}")
                forecast[extreme_low_mask] = lower_limit
            
            # 4. 轻度平滑处理 - 减少窗口大小，避免过度平滑
            forecast = pd.Series(forecast).rolling(window=3, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            
            # 5. 最终验证和微调
            final_mean = forecast.mean()
            final_bias = final_mean - data_mean
            
            if abs(final_bias) > data_std * 0.1:  # 如果还有较大偏差
                print(f"最终微调: 偏差 {final_bias:.2f}")
                # 线性调整到目标均值
                target_mean = data_mean + final_bias * 0.3  # 只校正70%的偏差，保留一些预测特性
                adjustment = target_mean - final_mean
                forecast = forecast + adjustment
                print(f"微调幅度: {adjustment:.2f}")
            
            print(f"最终预测值范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
            print(f"最终预测值均值: {forecast.mean():.2f} (目标: {data_mean:.2f})")
            print(f"最终偏差: {forecast.mean() - data_mean:.2f}")
            
            # 创建预测时间索引（从指定的预测日期开始）
            prediction_start_time = pd.Timestamp(f"{prediction_date} 00:00:00")
            prediction_index = pd.date_range(
                start=prediction_start_time,
                periods=len(forecast),
                freq='1min'
            )
            
            prediction_series = pd.Series(forecast, index=prediction_index)
            
            print(f"✅ Holt-Winters 预测完成: {len(prediction_series)} 个数据点")
            print(f"预测时间: {prediction_index[0]} 到 {prediction_index[-1]}")
            
            # 保存所使用的历史日期信息，供绘图使用
            prediction_series.prediction_info = {
                'prediction_date': prediction_date,
                'prediction_weekday_name': prediction_weekday_name,
                'training_dates': available_valid_dates,
                'two_weeks_data_date': two_weeks_data_date,
                'last_week_data_date': last_week_data_date
            }
            
            return prediction_series
            
        except Exception as e:
            print(f"Holt-Winters 预测失败: {e}")
            print("可能的原因: 数据量不足或季节性模式不明显")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_prediction_results(self, prediction_series, save_path='prediction_plot.png'):
        """
        绘制预测结果图：显示使用的历史数据和预测结果
        """
        print("正在绘制预测结果...")
        
        # 从prediction_series中获取预测信息
        if hasattr(prediction_series, 'prediction_info'):
            info = prediction_series.prediction_info
            prediction_date = info['prediction_date']
            prediction_weekday_name = info['prediction_weekday_name']
            two_weeks_data_date = info['two_weeks_data_date']
            last_week_data_date = info['last_week_data_date']
            
            print(f"使用保存的预测信息:")
            print(f"  预测日期: {prediction_date} ({prediction_weekday_name})")
            print(f"  使用的历史数据: {two_weeks_data_date}, {last_week_data_date}")
        else:
            # 如果没有保存的信息，使用默认逻辑（向后兼容）
            last_date = self.resampled_data.index.max().date()
            prediction_date = last_date + pd.Timedelta(days=1)
            prediction_weekday = pd.to_datetime(prediction_date).weekday()
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            prediction_weekday_name = weekday_names[prediction_weekday]
            
            print("⚠️  使用默认绘图逻辑（缺少预测信息）")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # 获取该工作日的历史数据
        weekday_data = self.weekday_patterns[prediction_weekday_name]
        
        if len(weekday_data) > 0 and hasattr(prediction_series, 'prediction_info'):
            # 使用具体的历史日期数据
            daily_data = weekday_data.groupby('date')
            
            # 获取使用的两天数据
            if two_weeks_data_date in daily_data.groups and last_week_data_date in daily_data.groups:
                last_week_data = daily_data.get_group(last_week_data_date)
                two_weeks_ago_data = daily_data.get_group(two_weeks_data_date)
                
                # 创建x轴标签（24小时）
                hours = list(range(24))
                
                # 绘制第一部分历史数据（较早日期）
                two_weeks_values = []
                two_weeks_sorted = two_weeks_ago_data.sort_values(['hour', 'minute'])
                
                for hour in range(24):
                    hour_data = two_weeks_sorted[two_weeks_sorted['hour'] == hour]
                    if len(hour_data) > 0:
                        hourly_avg = hour_data['value'].mean()
                        two_weeks_values.append(hourly_avg)
                    else:
                        two_weeks_values.append(None)
                
                # 绘制第二部分历史数据（较近日期）
                last_week_values = []
                last_week_sorted = last_week_data.sort_values(['hour', 'minute'])
                
                for hour in range(24):
                    hour_data = last_week_sorted[last_week_sorted['hour'] == hour]
                    if len(hour_data) > 0:
                        hourly_avg = hour_data['value'].mean()
                        last_week_values.append(hourly_avg)
                    else:
                        last_week_values.append(None)
                
                # 绘制预测数据（按小时平均）
                prediction_values = []
                prediction_df = pd.DataFrame({
                    'value': prediction_series.values,
                    'hour': prediction_series.index.hour
                })
                
                for hour in range(24):
                    hour_data = prediction_df[prediction_df['hour'] == hour]
                    if len(hour_data) > 0:
                        hourly_avg = hour_data['value'].mean()
                        prediction_values.append(hourly_avg)
                    else:
                        prediction_values.append(None)
                
                # 绘制三条线
                ax.plot(hours, two_weeks_values, 
                       color='blue', linewidth=2.5, marker='o', markersize=4,
                       label=f'历史数据1: {two_weeks_data_date} ({prediction_weekday_name})', alpha=0.8)
                
                ax.plot(hours, last_week_values, 
                       color='green', linewidth=2.5, marker='s', markersize=4,
                       label=f'历史数据2: {last_week_data_date} ({prediction_weekday_name})', alpha=0.8)
                
                # 预测日期
                ax.plot(hours, prediction_values, 
                       color='red', linewidth=3, marker='^', markersize=5,
                       label=f'预测: {prediction_date} ({prediction_weekday_name})', linestyle='--')
                
                # 设置x轴标签
                ax.set_xticks(range(0, 24, 2))  # 每2小时一个刻度
                ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 24, 2)])
                
                # 设置标题和标签
                ax.set_title(f'天然气生产预测对比 - {prediction_weekday_name} (Holt-Winters方法)', 
                           fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('时间 (小时)', fontsize=12)
                ax.set_ylabel('总系统瞬时（计算）', fontsize=12)
                
                # 添加网格
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.3, which='minor')
                
                # 设置图例
                ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
                
                # 添加数值标注（可选，在关键时段显示数值）
                key_hours = [0, 6, 12, 18, 23]  # 关键时间点
                for hour in key_hours:
                    if hour < len(prediction_values) and prediction_values[hour] is not None:
                        ax.annotate(f'{prediction_values[hour]:.1f}', 
                                  xy=(hour, prediction_values[hour]), 
                                  xytext=(5, 10), textcoords='offset points',
                                  fontsize=9, alpha=0.7, color='red',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                
                # 添加说明文本
                info_text = f"数据说明:\n• 历史数据1: {two_weeks_data_date}\n• 历史数据2: {last_week_data_date}\n• 预测日期: {prediction_date}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
            else:
                # 如果无法找到指定的历史数据，绘制简单的预测图
                ax.plot(prediction_series.index, prediction_series.values, 
                       color='red', linewidth=2, label='预测数据', linestyle='--')
                ax.set_title(f'天然气生产预测结果 - {prediction_date} ({prediction_weekday_name})', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('时间')
                ax.set_ylabel('总系统瞬时（计算）')
                ax.legend()
        else:
            # 如果历史数据不足或没有预测信息，绘制简单的预测图
            ax.plot(prediction_series.index, prediction_series.values, 
                   color='red', linewidth=2, label='预测数据', linestyle='--')
            ax.set_title('天然气生产预测结果', fontsize=14, fontweight='bold')
            ax.set_xlabel('时间')
            ax.set_ylabel('总系统瞬时（计算）')
            ax.legend()
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 确保保存路径存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存到: {os.path.abspath(save_path)}")
        plt.close()

def main():
    """主函数"""
    print("天然气预测工具 - 核心版本")
    print("=" * 40)
    
    # 创建预测器
    predictor = GasPredictor()
    
    # 处理数据
    predictor.load_and_process_data()
    
    # 询问用户是否要自定义预测日期
    print("\n📅 预测日期设置:")
    use_custom_date = input("是否使用自定义预测日期？(y/N): ").strip().lower()
    
    custom_prediction_date = None
    if use_custom_date in ['y', 'yes', '是']:
        while True:
            try:
                date_input = input("请输入预测日期 (格式: YYYY-MM-DD，如 2025-07-07): ").strip()
                if date_input:
                    # 验证日期格式
                    custom_prediction_date = pd.to_datetime(date_input).date()
                    
                    # 检查日期是否合理（不能是过去太久的日期）
                    today = pd.Timestamp.now().date()
                    if custom_prediction_date < today - pd.Timedelta(days=365):
                        print("⚠️  日期过于久远，请选择一个更近的日期")
                        continue
                    
                    print(f"✅ 将预测日期: {custom_prediction_date}")
                    break
                else:
                    print("❌ 日期不能为空，请重新输入")
            except Exception as e:
                print(f"❌ 日期格式错误: {e}")
                print("请使用格式: YYYY-MM-DD (如: 2025-07-07)")
    else:
        print("使用默认预测日期 (最新数据后一天)")
    
    # 进行预测
    print(f"\n🔮 开始预测...")
    result = predictor.predict(prediction_days=1, custom_prediction_date=custom_prediction_date)
    
    if result is not None:
        print(f"\n✅ 预测成功!")
        print(f"预测值范围: {result.min():.2f} ~ {result.max():.2f}")
        print(f"预测值均值: {result.mean():.2f}")
        
        # 生成文件名（包含预测日期）
        if hasattr(result, 'prediction_info'):
            prediction_date = result.prediction_info['prediction_date']
            date_str = str(prediction_date).replace('-', '')
            csv_filename = f'prediction_{date_str}.csv'
            plot_filename = f'prediction_{date_str}.png'
        else:
            csv_filename = 'prediction_result.csv'
            plot_filename = 'prediction_plot.png'
        
        # 保存CSV结果
        result.to_csv(csv_filename, header=['predicted_value'])
        print(f"📊 结果已保存到: {csv_filename}")
        
        # 绘制并保存图片
        predictor.plot_prediction_results(result, plot_filename)
        
        # 显示预测信息摘要
        if hasattr(result, 'prediction_info'):
            info = result.prediction_info
            print(f"\n📈 预测摘要:")
            print(f"   预测日期: {info['prediction_date']} ({info['prediction_weekday_name']})")
            print(f"   使用的历史数据:")
            print(f"     - 数据1: {info['two_weeks_data_date']}")
            print(f"     - 数据2: {info['last_week_data_date']}")
            print(f"   预测时间段: {result.index[0].strftime('%Y-%m-%d %H:%M')} ~ {result.index[-1].strftime('%Y-%m-%d %H:%M')}")
        
    else:
        print("❌ 预测失败")
        print("可能原因:")
        print("  1. 指定日期没有足够的历史同工作日数据")
        print("  2. 历史数据质量不佳")
        print("  3. 模型参数不适合当前数据")

if __name__ == "__main__":
    main() 
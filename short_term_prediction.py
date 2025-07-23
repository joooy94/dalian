#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns

warnings.filterwarnings('ignore')

class ShortTermPredictor:
    """短期气体流量预测器 - 使用7天数据预测特定日期（增强版：工作日/周末差异化）"""
    
    def __init__(self, data_file='历史数据2_total_flow.csv', 
                 model_dir='models', 
                 output_dir='short_predictions',
                 viz_dir='visualizations'):
        
        self.data_file = data_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.viz_dir = viz_dir
        
        # 创建目录
        for directory in [self.model_dir, self.output_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.df = None
        self.model = None
        self.weekend_model = None  # 周末专用模型
        
        # 设置英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self, training_days=7):
        """加载数据并选择最近N天作为训练数据，计算工作日/周末差异"""
        print(f"正在加载数据，使用最近{training_days}天作为训练数据...")
        
        self.df = pd.read_csv(self.data_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # 添加时间特征
        self.df['hour'] = self.df['time'].dt.hour
        self.df['day_of_week'] = self.df['time'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # 选择最近N天的数据
        latest_date = self.df['time'].max()
        start_date = latest_date - timedelta(days=training_days)
        
        self.training_data = self.df[self.df['time'] >= start_date].copy()
        
        # 计算工作日和周末的平均流量差异
        weekday_data = self.df[self.df['is_weekend'] == 0]['total_flow']
        weekend_data = self.df[self.df['is_weekend'] == 1]['total_flow']
        
        self.weekday_avg = weekday_data.mean() if len(weekday_data) > 0 else 0
        self.weekend_avg = weekend_data.mean() if len(weekend_data) > 0 else 0
        self.weekend_ratio = self.weekend_avg / self.weekday_avg if self.weekday_avg > 0 else 0.8
        
        # 分离训练数据中的工作日和周末数据
        self.training_weekday_data = self.training_data[self.training_data['is_weekend'] == 0]
        self.training_weekend_data = self.training_data[self.training_data['is_weekend'] == 1]
        
        print(f"训练数据: {len(self.training_data)}条记录")
        print(f"时间范围: {self.training_data['time'].min()} 到 {self.training_data['time'].max()}")
        print(f"训练数据平均流量: {self.training_data['total_flow'].mean():.2f}")
        print(f"工作日平均流量: {self.weekday_avg:.2f}")
        print(f"周末平均流量: {self.weekend_avg:.2f}")
        print(f"周末/工作日比例: {self.weekend_ratio:.3f}")
        print(f"训练数据中工作日记录: {len(self.training_weekday_data)}")
        print(f"训练数据中周末记录: {len(self.training_weekend_data)}")
        return self
    
    def get_historical_data_for_date(self, target_date):
        """获取指定日期的真实历史数据"""
        target_date_only = target_date.date()
        mask = (self.df['time'].dt.date == target_date_only)
        historical_data = self.df[mask].copy()
        
        if len(historical_data) > 0:
            # 添加必要的时间特征
            historical_data['hour'] = historical_data['time'].dt.hour
            historical_data['day_of_week'] = historical_data['time'].dt.dayofweek
            return historical_data[['time', 'total_flow', 'hour', 'day_of_week']].copy()
        
        # 如果没有完全匹配的日期，寻找相同星期几的数据
        target_weekday = target_date.weekday()
        self.df['day_of_week'] = self.df['time'].dt.dayofweek  # 确保有这个列
        weekday_mask = (self.df['day_of_week'] == target_weekday)
        weekday_data = self.df[weekday_mask].copy()
        
        if len(weekday_data) > 0:
            # 在训练数据范围内查找相同星期几的数据
            training_start = self.training_data['time'].min()
            recent_weekday_data = weekday_data[weekday_data['time'] >= training_start]
            
            if len(recent_weekday_data) > 0:
                unique_dates = recent_weekday_data['time'].dt.date.unique()
                if len(unique_dates) > 0:
                    latest_date = max(unique_dates)
                    sample_date_data = recent_weekday_data[recent_weekday_data['time'].dt.date == latest_date].copy()
                    # 添加必要的时间特征
                    sample_date_data['hour'] = sample_date_data['time'].dt.hour
                    sample_date_data['day_of_week'] = sample_date_data['time'].dt.dayofweek
                    return sample_date_data[['time', 'total_flow', 'hour', 'day_of_week']].copy()
        
        return None

    def train_holt_winters(self):
        """训练Holt-Winters模型（支持工作日/周末分离建模）"""
        print("训练 Holt-Winters 模型（基于7天数据，支持工作日/周末差异化）...")
        
        try:
            # 使用所有训练数据训练基础模型
            flow_data = self.training_data['total_flow']
            
            # 根据数据量调整季节周期
            seasonal_periods = 144  # 1天的周期
            
            # 如果数据量不够，调整季节周期
            if len(flow_data) < seasonal_periods * 2:
                seasonal_periods = max(24, len(flow_data) // 3)  # 至少4小时，或数据长度的1/3
                print(f"调整季节周期为: {seasonal_periods}")
            
            # 训练主模型（基于全部数据）
            self.model = ExponentialSmoothing(
                flow_data,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit(optimized=True)
            
            # 尝试训练周末专用模型
            self.has_separate_weekend_model = False
            weekend_model_success = False
            
            if len(self.training_weekend_data) >= 288:  # 至少两天的周末数据
                print("训练周末专用Holt-Winters模型...")
                try:
                    weekend_flow_data = self.training_weekend_data['total_flow']
                    weekend_seasonal_periods = min(144, len(weekend_flow_data) // 3)  # 更保守的周期设置
                    
                    # 检查周末数据的统计特性
                    weekend_std = weekend_flow_data.std()
                    weekend_mean = weekend_flow_data.mean()
                    
                    print(f"周末数据统计: 均值={weekend_mean:.2f}, 标准差={weekend_std:.2f}")
                    
                    # 只有在数据质量良好时才训练周末专用模型
                    if weekend_std > 0 and weekend_mean > 0:
                        self.weekend_model = ExponentialSmoothing(
                            weekend_flow_data,
                            trend='add',
                            seasonal='add',
                            seasonal_periods=weekend_seasonal_periods
                        ).fit(optimized=True)
                        
                        # 验证模型预测效果
                        test_forecast = self.weekend_model.forecast(steps=12)  # 测试2小时预测
                        
                        # 检查预测值是否合理（无负值，与历史数据范围一致）
                        if np.any(test_forecast < 0) or np.any(test_forecast > weekend_mean * 3):
                            print("周末模型预测值异常，使用主模型+修正因子方案")
                            self.weekend_model = None
                        else:
                            self.has_separate_weekend_model = True
                            weekend_model_success = True
                            print(f"周末专用模型训练成功，季节周期: {weekend_seasonal_periods}")
                    else:
                        print("周末数据质量不佳，跳过专用模型训练")
                        
                except Exception as e:
                    print(f"周末专用模型训练失败: {e}")
                    self.weekend_model = None
            else:
                print("周末数据不足（需要至少288个点），将使用主模型 + 周末修正因子")
            
            if not weekend_model_success:
                print("最终策略：使用主模型 + 周末修正因子的方式")
                self.weekend_model = None
                self.has_separate_weekend_model = False
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'enhanced_holt_winters.pkl')
            models_to_save = {
                'main_model': self.model,
                'weekend_model': self.weekend_model,
                'has_separate_weekend_model': self.has_separate_weekend_model,
                'weekday_avg': self.weekday_avg,
                'weekend_avg': self.weekend_avg,
                'weekend_ratio': self.weekend_ratio,
                'seasonal_periods': seasonal_periods
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(models_to_save, f)
            
            # 保存模型信息
            model_info = f"""Enhanced Holt-Winters Model with Weekend/Weekday Differentiation
Training Data: {len(flow_data)} points ({len(flow_data)/144:.1f} days)
Main Model Seasonal Periods: {seasonal_periods}
Time Range: {self.training_data['time'].min()} to {self.training_data['time'].max()}
Average Flow: {flow_data.mean():.2f}
Weekday Average: {self.weekday_avg:.2f}
Weekend Average: {self.weekend_avg:.2f}
Weekend/Weekday Ratio: {self.weekend_ratio:.3f}
Separate Weekend Model: {self.has_separate_weekend_model}
Weekend Training Data Points: {len(self.training_weekend_data)}
Model Strategy: {'Weekend-specific Model' if self.has_separate_weekend_model else 'Main Model + Ratio Correction'}
"""
            info_path = os.path.join(self.model_dir, 'enhanced_holt_winters_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(model_info)
            
            print(f"增强版Holt-Winters模型已保存到 {model_path}")
            print(f"工作日/周末差异化处理：周末修正比例 {self.weekend_ratio:.3f}")
            return True
            
        except Exception as e:
            print(f"Holt-Winters 训练失败: {e}")
            return False
    
    def predict_single_day(self, target_date):
        """预测单个日期的24小时数据（支持工作日/周末差异化）"""
        if self.model is None:
            print("模型未训练，请先调用 train_holt_winters()")
            return None
        
        try:
            # 判断目标日期是否为周末
            is_weekend = target_date.weekday() >= 5
            day_type = "周末" if is_weekend else "工作日"
            
            print(f"  预测 {target_date} ({day_type})，使用{'周末专用模型' if (is_weekend and self.has_separate_weekend_model) else '主模型' + ('+ 周末修正' if is_weekend else '')}")
            
            # 计算从训练数据结束到目标日期的步数
            last_training_time = self.training_data['time'].max()
            target_start = datetime.combine(target_date, datetime.min.time())
            
            # 选择使用的模型和处理方式
            if is_weekend and self.has_separate_weekend_model:
                # 使用周末专用模型
                selected_model = self.weekend_model
                use_ratio_correction = False
                print("  使用周末专用Holt-Winters模型")
            else:
                # 使用主模型
                selected_model = self.model
                use_ratio_correction = is_weekend  # 周末时需要应用修正比例
                if is_weekend:
                    print(f"  使用主模型 + 周末修正因子 ({self.weekend_ratio:.3f})")
            
            # 如果目标日期在训练数据范围内或之前，我们需要特殊处理
            if target_start <= last_training_time:
                print(f"  目标日期在训练数据范围内，使用模型拟合值")
                
                try:
                    # 预测未来24小时
                    forecast = selected_model.forecast(steps=144)  # 24小时 = 144个10分钟间隔
                    
                    # 创建时间序列
                    time_range = pd.date_range(
                        start=target_start, 
                        periods=144, 
                        freq='10T'
                    )
                    
                    pred_df = pd.DataFrame({
                        'time': time_range,
                        'predicted_flow': forecast
                    })
                    
                except:
                    # 如果直接预测失败，使用训练数据的同一天数据作为基准
                    target_day_of_week = target_date.weekday()
                    
                    if is_weekend and len(self.training_weekend_data) > 0:
                        # 使用周末数据
                        same_weekday_data = self.training_weekend_data[
                            self.training_weekend_data['day_of_week'] == target_day_of_week
                        ]
                    else:
                        # 使用工作日数据或全部数据
                        same_weekday_data = self.training_data[
                            self.training_data['day_of_week'] == target_day_of_week
                        ]
                    
                    if len(same_weekday_data) > 0:
                        # 取最近的同一星期几的数据作为预测基础
                        latest_same_day = same_weekday_data['time'].dt.date.max()
                        latest_day_data = same_weekday_data[
                            same_weekday_data['time'].dt.date == latest_same_day
                        ]
                        
                        if len(latest_day_data) >= 144:
                            time_range = pd.date_range(
                                start=target_start, 
                                periods=144, 
                                freq='10T'
                            )
                            
                            pred_df = pd.DataFrame({
                                'time': time_range,
                                'predicted_flow': latest_day_data['total_flow'].values[:144]
                            })
                        else:
                            # 使用对应数据类型的平均值
                            if is_weekend:
                                avg_flow = self.weekend_avg if self.weekend_avg > 0 else self.training_data['total_flow'].mean()
                            else:
                                avg_flow = self.weekday_avg if self.weekday_avg > 0 else self.training_data['total_flow'].mean()
                            
                            time_range = pd.date_range(
                                start=target_start, 
                                periods=144, 
                                freq='10T'
                            )
                            
                            pred_df = pd.DataFrame({
                                'time': time_range,
                                'predicted_flow': [avg_flow] * 144
                            })
                    else:
                        # 使用整体平均值
                        avg_flow = self.training_data['total_flow'].mean()
                        time_range = pd.date_range(
                            start=target_start, 
                            periods=144, 
                            freq='10T'
                        )
                        
                        pred_df = pd.DataFrame({
                            'time': time_range,
                            'predicted_flow': [avg_flow] * 144
                        })
                        
            else:
                # 目标日期在未来，进行正常预测
                print(f"  目标日期在未来，进行前向预测")
                
                # 计算需要预测的总步数（到目标日期开始 + 24小时）
                time_diff = target_start - last_training_time
                steps_to_target = int(time_diff.total_seconds() / 600)  # 10分钟间隔
                total_steps = steps_to_target + 144  # 24小时 = 144个10分钟间隔
                
                # 进行预测
                forecast = selected_model.forecast(steps=total_steps)
                
                # 提取目标日期的24小时预测
                target_predictions = forecast[steps_to_target:steps_to_target + 144]
                
                # 创建时间序列
                time_range = pd.date_range(
                    start=target_start, 
                    periods=144, 
                    freq='10T'
                )
                
                pred_df = pd.DataFrame({
                    'time': time_range,
                    'predicted_flow': target_predictions
                })
            
            # 应用周末修正因子（如果需要）
            if use_ratio_correction:
                print(f"  应用周末修正因子: {self.weekend_ratio:.3f}")
                pred_df['predicted_flow'] = pred_df['predicted_flow'] * self.weekend_ratio
            
            # 验证预测结果并应用安全检查
            if 'predicted_flow' in pred_df.columns:
                # 检查负值
                negative_count = (pred_df['predicted_flow'] < 0).sum()
                if negative_count > 0:
                    print(f"  Warning: 发现 {negative_count} 个负值预测，应用修正...")
                    
                    # 使用对应类型的历史平均值替换负值
                    if is_weekend:
                        replacement_value = max(self.weekend_avg, 0)
                    else:
                        replacement_value = max(self.weekday_avg, 0)
                    
                    # 如果历史平均值也是0或负数，使用训练数据的最小正值
                    if replacement_value <= 0:
                        min_positive = self.training_data[self.training_data['total_flow'] > 0]['total_flow'].min()
                        replacement_value = min_positive if not np.isnan(min_positive) else 10.0
                    
                    pred_df.loc[pred_df['predicted_flow'] < 0, 'predicted_flow'] = replacement_value
                    print(f"  负值已替换为: {replacement_value:.2f}")
                
                # 检查异常高值
                mean_val = pred_df['predicted_flow'].mean()
                std_val = pred_df['predicted_flow'].std()
                upper_bound = mean_val + 4 * std_val  # 4倍标准差
                high_count = (pred_df['predicted_flow'] > upper_bound).sum()
                
                if high_count > 0:
                    print(f"  Warning: 发现 {high_count} 个异常高值，应用修正...")
                    pred_df.loc[pred_df['predicted_flow'] > upper_bound, 'predicted_flow'] = upper_bound
                
                # 最终验证：确保所有预测值都是合理的正数
                pred_df['predicted_flow'] = np.maximum(pred_df['predicted_flow'], 0.1)
            
            # 添加时间特征
            pred_df['hour'] = pred_df['time'].dt.hour
            pred_df['day_of_week'] = pred_df['time'].dt.dayofweek
            pred_df['is_weekend'] = (pred_df['day_of_week'] >= 5).astype(int)
            
            return pred_df
            
        except Exception as e:
            print(f"预测日期 {target_date} 失败: {e}")
            return None
    
    def run_target_predictions(self):
        """预测指定的目标日期"""
        target_dates = [
            datetime(2025, 6, 11).date(),  # 6月11日
            datetime(2025, 6, 14).date(),  # 6月14日
            datetime(2025, 6, 18).date(),  # 6月18日
            datetime(2025, 6, 19).date(),  # 6月19日
        ]
        
        print("\n=== 开始增强版Holt-Winters预测目标日期 ===")
        results = {}
        
        for target_date in target_dates:
            print(f"\n预测日期: {target_date}")
            
            predictions = self.predict_single_day(target_date)
            if predictions is not None:
                avg_flow = predictions['predicted_flow'].mean()
                max_flow = predictions['predicted_flow'].max()
                min_flow = predictions['predicted_flow'].min()
                
                print(f"  平均流量: {avg_flow:.2f}")
                print(f"  最大流量: {max_flow:.2f}")
                print(f"  最小流量: {min_flow:.2f}")
                
                results[target_date] = predictions
                
                # 保存预测结果
                date_str = target_date.strftime("%Y%m%d")
                filename = f"enhanced_prediction_{date_str}.csv"
                filepath = os.path.join(self.output_dir, filename)
                predictions.to_csv(filepath, index=False, encoding='utf-8')
                print(f"  已保存到: {filename}")
        
        return results
    
    def create_prediction_visualization(self, all_results):
        """创建预测结果可视化（包含历史数据对比）"""
        print("\n=== 生成增强版Holt-Winters预测结果对比图表 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Holt-Winters: Weekend/Weekday Differentiated Predictions vs Historical Data', fontsize=16, fontweight='bold')
        
        target_dates = list(all_results.keys())
        
        for idx, target_date in enumerate(target_dates):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            predictions = all_results[target_date]
            is_weekend = target_date.weekday() >= 5
            
            # 选择颜色
            pred_color = '#ff6b6b' if is_weekend else '#4ecdc4'  # 周末用红色，工作日用青色
            
            # 绘制预测数据
            model_type = f"Enhanced HW ({'Weekend Model' if (is_weekend and self.has_separate_weekend_model) else 'Main + Correction' if is_weekend else 'Main Model'})"
            ax.plot(predictions['time'], predictions['predicted_flow'], 
                   label=model_type, color=pred_color, 
                   linewidth=2.5, alpha=0.9)
            
            # 获取并绘制历史数据
            historical_data = self.get_historical_data_for_date(datetime.combine(target_date, datetime.min.time()))
            if historical_data is not None and len(historical_data) > 0:
                ax.plot(historical_data['time'], historical_data['total_flow'], 
                       label='Historical Data', color='#1f77b4', 
                       linewidth=2, alpha=0.8, linestyle='--')
            
            # 设置图表
            date_str = target_date.strftime("%B %d, %Y")
            weekday = target_date.strftime("%A")
            day_type = "Weekend" if is_weekend else "Weekday"
            ax.set_title(f'{date_str} ({weekday} - {day_type})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Gas Flow Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'enhanced_holt_winters_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"增强版可视化图表已保存到: {save_path}")
    
    def generate_prediction_report(self, all_results):
        """生成预测报告"""
        print("\n=== 生成增强版Holt-Winters预测报告 ===")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Enhanced Holt-Winters Gas Flow Prediction Report")
        report_lines.append("Weekend/Weekday Differentiated Predictions")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 训练数据概况
        report_lines.append("1. Training Data Overview")
        report_lines.append("-"*30)
        report_lines.append(f"Training Records: {len(self.training_data)}")
        report_lines.append(f"Training Period: {self.training_data['time'].min()} to {self.training_data['time'].max()}")
        report_lines.append(f"Training Days: {(self.training_data['time'].max() - self.training_data['time'].min()).days + 1}")
        report_lines.append(f"Average Training Flow: {self.training_data['total_flow'].mean():.2f}")
        report_lines.append(f"Training Flow Range: {self.training_data['total_flow'].min():.2f} - {self.training_data['total_flow'].max():.2f}")
        report_lines.append(f"Weekday Average: {self.weekday_avg:.2f}")
        report_lines.append(f"Weekend Average: {self.weekend_avg:.2f}")
        report_lines.append(f"Weekend/Weekday Ratio: {self.weekend_ratio:.3f}")
        report_lines.append(f"Separate Weekend Model: {self.has_separate_weekend_model}")
        report_lines.append("")
        
        # 各日期预测结果
        for idx, (target_date, predictions) in enumerate(all_results.items(), 2):
            date_str = target_date.strftime("%Y-%m-%d")
            weekday = target_date.strftime("%A")
            is_weekend = target_date.weekday() >= 5
            day_type = "Weekend" if is_weekend else "Weekday"
            
            report_lines.append(f"{idx}. Prediction for {date_str} ({weekday} - {day_type})")
            report_lines.append("-"*30)
            
            # 模型信息
            if is_weekend and self.has_separate_weekend_model:
                model_used = "Weekend-specific Holt-Winters Model"
            elif is_weekend:
                model_used = f"Main Holt-Winters + Weekend Correction ({self.weekend_ratio:.3f})"
            else:
                model_used = "Main Holt-Winters Model"
            
            report_lines.append(f"  Model Used: {model_used}")
            
            avg_flow = predictions['predicted_flow'].mean()
            max_flow = predictions['predicted_flow'].max()
            min_flow = predictions['predicted_flow'].min()
            
            # 时间段分析
            morning_mask = (predictions['hour'] >= 6) & (predictions['hour'] < 12)
            afternoon_mask = (predictions['hour'] >= 12) & (predictions['hour'] < 18)
            worktime_mask = (predictions['hour'] >= 8) & (predictions['hour'] <= 18)
            
            morning_avg = predictions[morning_mask]['predicted_flow'].mean() if morning_mask.any() else 0
            afternoon_avg = predictions[afternoon_mask]['predicted_flow'].mean() if afternoon_mask.any() else 0
            worktime_avg = predictions[worktime_mask]['predicted_flow'].mean() if worktime_mask.any() else 0
            non_worktime_avg = predictions[~worktime_mask]['predicted_flow'].mean() if (~worktime_mask).any() else 0
            
            worktime_diff = ((worktime_avg / non_worktime_avg - 1) * 100) if non_worktime_avg > 0 else 0
            
            report_lines.append(f"  Average Flow: {avg_flow:.2f}")
            report_lines.append(f"  Maximum Flow: {max_flow:.2f}")
            report_lines.append(f"  Minimum Flow: {min_flow:.2f}")
            report_lines.append(f"  Morning (6-12h): {morning_avg:.2f}")
            report_lines.append(f"  Afternoon (12-18h): {afternoon_avg:.2f}")
            report_lines.append(f"  Work vs Non-work Hours: {worktime_diff:.1f}%")
            
            # 与训练数据平均值的比较
            training_avg = self.training_data['total_flow'].mean()
            diff_from_training = ((avg_flow / training_avg - 1) * 100) if training_avg > 0 else 0
            
            # 与对应类型平均值的比较
            if is_weekend:
                type_avg = self.weekend_avg
                type_name = "Weekend"
            else:
                type_avg = self.weekday_avg
                type_name = "Weekday"
            
            diff_from_type = ((avg_flow / type_avg - 1) * 100) if type_avg > 0 else 0
            
            report_lines.append(f"  Difference from Training Average: {diff_from_training:.1f}%")
            report_lines.append(f"  Difference from {type_name} Average: {diff_from_type:.1f}%")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'enhanced_holt_winters_prediction_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"增强版报告已保存到: {report_path}")
        print("\n" + report_text)

def main():
    # 创建增强版短期预测器
    predictor = ShortTermPredictor()
    
    # 加载7天训练数据
    predictor.load_data(training_days=7)
    
    # 训练增强版模型
    if predictor.train_holt_winters():
        # 预测目标日期
        results = predictor.run_target_predictions()
        
        if results:
            # 生成可视化和报告
            predictor.create_prediction_visualization(results)
            predictor.generate_prediction_report(results)
            
            print("\n" + "="*60)
            print("增强版Holt-Winters预测完成！")
            print("="*60)
            print("生成的文件:")
            print(f"1. 模型文件: {predictor.model_dir}/")
            print(f"2. 预测结果: {predictor.output_dir}/")
            print(f"3. 可视化图表: {predictor.viz_dir}/enhanced_holt_winters_predictions.png")
            print(f"4. 分析报告: {predictor.output_dir}/enhanced_holt_winters_prediction_report.txt")
            print("\n特色功能:")
            print("✓ 工作日/周末自动识别")
            print("✓ 周末专用模型（数据充足时）")
            print("✓ 智能修正因子应用")
            print("✓ 差异化预测策略")
        else:
            print("预测失败，请检查数据和模型。")
    else:
        print("模型训练失败。")

if __name__ == "__main__":
    main() 
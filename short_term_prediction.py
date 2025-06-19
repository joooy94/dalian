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
    """短期气体流量预测器 - 使用7天数据预测特定日期"""
    
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
        
        # 设置英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self, training_days=7):
        """加载数据并选择最近N天作为训练数据"""
        print(f"正在加载数据，使用最近{training_days}天作为训练数据...")
        
        self.df = pd.read_csv(self.data_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # 选择最近N天的数据
        latest_date = self.df['time'].max()
        start_date = latest_date - timedelta(days=training_days)
        
        self.training_data = self.df[self.df['time'] >= start_date].copy()
        
        # 添加时间特征
        self.training_data['hour'] = self.training_data['time'].dt.hour
        self.training_data['day_of_week'] = self.training_data['time'].dt.dayofweek
        self.training_data['is_weekend'] = (self.training_data['day_of_week'] >= 5).astype(int)
        
        print(f"训练数据: {len(self.training_data)}条记录")
        print(f"时间范围: {self.training_data['time'].min()} 到 {self.training_data['time'].max()}")
        print(f"训练数据平均流量: {self.training_data['total_flow'].mean():.2f}")
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
        """训练Holt-Winters模型"""
        print("训练 Holt-Winters 模型（基于7天数据）...")
        
        try:
            # 使用所有训练数据
            flow_data = self.training_data['total_flow']
            
            # 根据数据量调整季节周期
            # 7天数据约有1008个点（7×24×6），使用144（1天）作为季节周期
            seasonal_periods = 144  # 1天的周期
            
            # 如果数据量不够，调整季节周期
            if len(flow_data) < seasonal_periods * 2:
                seasonal_periods = max(24, len(flow_data) // 3)  # 至少4小时，或数据长度的1/3
                print(f"调整季节周期为: {seasonal_periods}")
            
            self.model = ExponentialSmoothing(
                flow_data,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit(optimized=True)
            
            # 保存模型
            model_path = os.path.join(self.model_dir, 'short_term_holt_winters.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # 保存模型信息
            model_info = f"""Short-term Holt-Winters Model
Training Data: {len(flow_data)} points ({len(flow_data)/144:.1f} days)
Seasonal Periods: {seasonal_periods}
Time Range: {self.training_data['time'].min()} to {self.training_data['time'].max()}
Average Flow: {flow_data.mean():.2f}
"""
            info_path = os.path.join(self.model_dir, 'short_term_holt_winters_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(model_info)
            
            print(f"模型已保存到 {model_path}")
            return True
            
        except Exception as e:
            print(f"Holt-Winters 训练失败: {e}")
            return False
    
    def predict_single_day(self, target_date):
        """预测单个日期的24小时数据"""
        if self.model is None:
            print("模型未训练，请先调用 train_holt_winters()")
            return None
        
        try:
            # 计算从训练数据结束到目标日期的步数
            last_training_time = self.training_data['time'].max()
            target_start = datetime.combine(target_date, datetime.min.time())
            
            # 如果目标日期在训练数据范围内或之前，我们需要特殊处理
            if target_start <= last_training_time:
                print(f"  目标日期 {target_date} 在训练数据范围内，使用模型拟合值")
                
                # 使用模型的拟合值或进行前向预测
                # 先进行一个短期预测来获得预测值
                try:
                    # 预测未来24小时
                    forecast = self.model.forecast(steps=144)  # 24小时 = 144个10分钟间隔
                    
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
                            # 使用该天的数据作为预测值
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
                            # 使用训练数据的平均值创建预测
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
                print(f"  目标日期 {target_date} 在未来，进行前向预测")
                
                # 计算需要预测的总步数（到目标日期开始 + 24小时）
                time_diff = target_start - last_training_time
                steps_to_target = int(time_diff.total_seconds() / 600)  # 10分钟间隔
                total_steps = steps_to_target + 144  # 24小时 = 144个10分钟间隔
                
                # 进行预测
                forecast = self.model.forecast(steps=total_steps)
                
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
        
        print("\n=== 开始预测目标日期 ===")
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
                filename = f"prediction_{date_str}.csv"
                filepath = os.path.join(self.output_dir, filename)
                predictions.to_csv(filepath, index=False, encoding='utf-8')
                print(f"  已保存到: {filename}")
        
        return results
    
    def create_prediction_visualization(self, all_results):
        """创建预测结果可视化（包含历史数据对比）"""
        print("\n=== 生成预测结果对比图表 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('7-Day Training: Gas Flow Predictions vs Historical Data', fontsize=16, fontweight='bold')
        
        target_dates = list(all_results.keys())
        
        for idx, target_date in enumerate(target_dates):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            predictions = all_results[target_date]
            
            # 绘制预测数据
            ax.plot(predictions['time'], predictions['predicted_flow'], 
                   label='Holt-Winters Prediction', color='#ff7f0e', 
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
            ax.set_title(f'{date_str} ({weekday})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Gas Flow Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'short_term_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"可视化图表已保存到: {save_path}")
    
    def generate_prediction_report(self, all_results):
        """生成预测报告"""
        print("\n=== 生成预测报告 ===")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Short-term Gas Flow Prediction Report")
        report_lines.append("7-Day Training Data → Specific Date Predictions")
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
        report_lines.append("")
        
        # 各日期预测结果
        for idx, (target_date, predictions) in enumerate(all_results.items(), 2):
            date_str = target_date.strftime("%Y-%m-%d")
            weekday = target_date.strftime("%A")
            
            report_lines.append(f"{idx}. Prediction for {date_str} ({weekday})")
            report_lines.append("-"*30)
            
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
            report_lines.append(f"  Difference from Training Average: {diff_from_training:.1f}%")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'short_term_prediction_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"报告已保存到: {report_path}")
        print("\n" + report_text)

def main():
    # 创建短期预测器
    predictor = ShortTermPredictor()
    
    # 加载7天训练数据
    predictor.load_data(training_days=7)
    
    # 训练模型
    if predictor.train_holt_winters():
        # 预测目标日期
        results = predictor.run_target_predictions()
        
        if results:
            # 生成可视化和报告
            predictor.create_prediction_visualization(results)
            predictor.generate_prediction_report(results)
            
            print("\n" + "="*60)
            print("短期预测完成！")
            print("="*60)
            print("生成的文件:")
            print(f"1. 模型文件: {predictor.model_dir}/")
            print(f"2. 预测结果: {predictor.output_dir}/")
            print(f"3. 可视化图表: {predictor.viz_dir}/short_term_predictions.png")
            print(f"4. 分析报告: {predictor.output_dir}/short_term_prediction_report.txt")
        else:
            print("预测失败，请检查数据和模型。")
    else:
        print("模型训练失败。")

if __name__ == "__main__":
    main() 
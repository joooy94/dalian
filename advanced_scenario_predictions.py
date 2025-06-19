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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

warnings.filterwarnings('ignore')

class AdvancedScenarioPredictor:
    """高级气体流量场景预测器"""
    
    def __init__(self, data_file='历史数据2_total_flow.csv', 
                 model_dir='models', 
                 output_dir='test_predictions',
                 viz_dir='visualizations'):
        
        self.data_file = data_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.viz_dir = viz_dir
        
        # 创建目录
        for directory in [self.model_dir, self.output_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.df = None
        self.models = {}
        
        # 设置英文字体，避免中文字体问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        self.df = pd.read_csv(self.data_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # 添加时间特征
        self.df['hour'] = self.df['time'].dt.hour
        self.df['day_of_week'] = self.df['time'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_worktime'] = ((self.df['hour'] >= 8) & (self.df['hour'] <= 18)).astype(int)
        
        print(f"数据加载完成: {len(self.df)}条记录")
        print(f"时间范围: {self.df['time'].min()} 到 {self.df['time'].max()}")
        return self
    
    def get_historical_data_for_comparison(self, start_time, duration_hours=24):
        """获取对比用的历史真实数据"""
        end_time = start_time + timedelta(hours=duration_hours)
        
        # 查找匹配的历史数据（同一天的数据）
        mask = (self.df['time'].dt.date == start_time.date())
        historical_data = self.df[mask].copy()
        
        if len(historical_data) > 0:
            return historical_data[['time', 'total_flow', 'hour', 'day_of_week', 'is_weekend']]
        
        # 如果没有完全匹配的日期，寻找相同星期几的数据
        target_weekday = start_time.weekday()
        weekday_mask = (self.df['day_of_week'] == target_weekday)
        weekday_data = self.df[weekday_mask]
        
        if len(weekday_data) > 0:
            # 取最近的相同星期几的数据
            unique_dates = weekday_data['time'].dt.date.unique()
            if len(unique_dates) > 0:
                latest_date = max(unique_dates)
                sample_date_data = weekday_data[weekday_data['time'].dt.date == latest_date]
                return sample_date_data[['time', 'total_flow', 'hour', 'day_of_week', 'is_weekend']].copy()
        
        return None

    def create_features(self, data):
        """创建特征工程"""
        df = data.copy()
        
        # 滞后特征
        for lag in [1, 6, 12, 24]:  # 10分钟, 1小时, 2小时, 4小时前
            df[f'lag_{lag}'] = df['total_flow'].shift(lag)
        
        # 移动平均特征
        for window in [6, 12, 24]:  # 1小时, 2小时, 4小时移动平均
            df[f'ma_{window}'] = df['total_flow'].rolling(window=window).mean()
        
        # 周期性特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.dropna()
    
    def save_model(self, model, model_name, model_info=None):
        """保存模型"""
        model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        if model_info:
            info_path = os.path.join(self.model_dir, f'{model_name}_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(model_info)
        
        print(f"模型 {model_name} 已保存到 {model_path}")
    
    def load_model(self, model_name):
        """加载模型"""
        model_path = os.path.join(self.model_dir, f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def train_holt_winters(self, train_data, seasonal_periods=144):
        """训练Holt-Winters模型"""
        print("训练 Holt-Winters 模型...")
        
        try:
            # 使用最近的数据进行训练，确保足够的季节周期
            min_data_points = seasonal_periods * 4  # 至少4个完整周期
            if len(train_data) < min_data_points:
                recent_data = train_data['total_flow']
                print(f"Warning: Limited data ({len(recent_data)} points). Adjusting seasonal periods.")
                seasonal_periods = max(24, len(recent_data) // 4)  # 至少24点（4小时），或数据长度的1/4
            else:
                recent_data = train_data.tail(min_data_points)['total_flow']
            
            model = ExponentialSmoothing(
                recent_data,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            ).fit(optimized=True)
            
            # 保存模型
            model_info = f"Holt-Winters Model\nSeasonal periods: {seasonal_periods}\nTrained on: {len(recent_data)} samples\n"
            self.save_model(model, 'holt_winters', model_info)
            self.models['holt_winters'] = model
            
            return model
            
        except Exception as e:
            print(f"Holt-Winters 训练失败: {e}")
            return None
    
    def train_all_models(self):
        """训练所有模型（仅Holt-Winters）"""
        print("\n=== 开始训练模型 ===")
        
        # 使用80%的数据进行训练
        train_size = int(len(self.df) * 0.9)  # 增加到90%获得更多训练数据
        train_data = self.df.iloc[:train_size]
        
        print(f"使用前{train_size}条记录进行训练")
        
        # 只训练Holt-Winters模型
        self.train_holt_winters(train_data)
        
        available_models = len([m for m in self.models.values() if m is not None])
        print(f"完成！共有 {available_models} 个可用模型")

    def predict_with_holt_winters(self, start_time, duration_hours=24):
        """使用Holt-Winters预测"""
        model = self.load_model('holt_winters') or self.models.get('holt_winters')
        if model is None:
            return None
        
        # 预测步数
        steps = duration_hours * 6  # 每小时6个点（10分钟间隔）
        
        try:
            forecast = model.forecast(steps=steps)
            
            # 创建时间序列
            end_time = start_time + timedelta(hours=duration_hours)
            time_range = pd.date_range(start=start_time, end=end_time, freq='10T')[:-1]
            
            pred_df = pd.DataFrame({
                'time': time_range,
                'predicted_flow': forecast[:len(time_range)]
            })
            
            pred_df['hour'] = pred_df['time'].dt.hour
            pred_df['day_of_week'] = pred_df['time'].dt.dayofweek
            pred_df['is_weekend'] = (pred_df['day_of_week'] >= 5).astype(int)
            
            return pred_df
            
        except Exception as e:
            print(f"Holt-Winters 预测失败: {e}")
            return None
    
    def run_scenario_test(self, scenario_name, start_time, duration_hours=24):
        """运行场景测试"""
        print(f"\n{'='*50}")
        print(f"执行场景: {scenario_name}")
        print(f"{'='*50}")
        
        print(f"\n=== {scenario_name}预测 ===")
        print(f"预测时间: {start_time}")
        print(f"预测时长: {duration_hours}小时")
        
        results = {}
        
        # 只使用Holt-Winters模型进行预测
        try:
            predictions = self.predict_with_holt_winters(start_time, duration_hours)
            if predictions is not None:
                avg_flow = predictions['predicted_flow'].mean()
                print(f"Holt-Winters: 平均流量 {avg_flow:.2f}")
                results['holt_winters'] = predictions
                
                # 保存预测结果
                filename = f"{scenario_name}_holt_winters_predictions.csv"
                filepath = os.path.join(self.output_dir, filename)
                predictions.to_csv(filepath, index=False, encoding='utf-8')
            
        except Exception as e:
            print(f"Holt-Winters 预测失败: {e}")
        
        return results
    
    def analyze_predictions(self, predictions):
        """分析预测结果"""
        analysis = {}
        
        flow = predictions['predicted_flow']
        analysis['avg_flow'] = flow.mean()
        analysis['max_flow'] = flow.max()
        analysis['min_flow'] = flow.min()
        
        # 时间段分析
        predictions_copy = predictions.copy()
        morning_mask = (predictions_copy['hour'] >= 6) & (predictions_copy['hour'] < 12)
        afternoon_mask = (predictions_copy['hour'] >= 12) & (predictions_copy['hour'] < 18)
        worktime_mask = (predictions_copy['hour'] >= 8) & (predictions_copy['hour'] <= 18)
        
        analysis['morning_avg'] = predictions_copy[morning_mask]['predicted_flow'].mean() if morning_mask.any() else 0
        analysis['afternoon_avg'] = predictions_copy[afternoon_mask]['predicted_flow'].mean() if afternoon_mask.any() else 0
        analysis['worktime_avg'] = predictions_copy[worktime_mask]['predicted_flow'].mean() if worktime_mask.any() else 0
        analysis['non_worktime_avg'] = predictions_copy[~worktime_mask]['predicted_flow'].mean() if (~worktime_mask).any() else 0
        
        if analysis['non_worktime_avg'] > 0:
            analysis['worktime_increase'] = (analysis['worktime_avg'] / analysis['non_worktime_avg'] - 1) * 100
        else:
            analysis['worktime_increase'] = 0
        
        return analysis
    
    def create_scenario_comparison(self, all_results):
        """创建场景对比图（包含真实数据对比）"""
        print("\n=== 生成场景预测对比图表 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gas Flow Scenario Prediction vs Historical Data', fontsize=16, fontweight='bold')
        
        scenarios = list(all_results.keys())
        
        # 绘制每个场景的预测结果和真实数据对比
        for idx, (scenario, results) in enumerate(all_results.items()):
            row = idx // 2
            col = idx % 2
            
            if row < 2 and col < 2:
                ax = axes[row, col]
                
                # 获取对应场景的开始时间
                scenario_times = {
                    'Today_Jun19': datetime(2025, 6, 19, 0, 0),
                    'Weekend_Prediction': datetime(2025, 6, 21, 0, 0),
                    'Weekday_Prediction': datetime(2025, 6, 23, 0, 0),
                }
                
                start_time = scenario_times.get(scenario, datetime(2025, 6, 19, 0, 0))
                
                # 获取历史数据进行对比
                historical_data = self.get_historical_data_for_comparison(start_time, 24)
                
                # 绘制预测数据
                for model_name, predictions in results.items():
                    if predictions is not None:
                        ax.plot(predictions['time'], predictions['predicted_flow'], 
                               label=f'Holt-Winters Prediction', color='#ff7f0e', 
                               linewidth=2.5, alpha=0.9)
                
                # 绘制历史真实数据（如果有的话）
                if historical_data is not None and len(historical_data) > 0:
                    ax.plot(historical_data['time'], historical_data['total_flow'], 
                           label='Historical Data', color='#1f77b4', 
                           linewidth=2, alpha=0.8, linestyle='--')
                
                # 设置图表标题和标签
                scenario_titles = {
                    'Today_Jun19': 'Today (June 19, 2025)',
                    'Weekend_Prediction': 'Weekend Prediction (June 21, 2025)', 
                    'Weekday_Prediction': 'Weekday Prediction (June 23, 2025)'
                }
                
                ax.set_title(scenario_titles.get(scenario, scenario), fontsize=12, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Gas Flow Rate')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 格式化x轴时间显示
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 隐藏多余的子图
        if len(all_results) < 4:
            for idx in range(len(all_results), 4):
                row = idx // 2
                col = idx % 2
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'holt_winters_scenario_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"对比图表已保存到 {save_path}")
    
    def generate_comprehensive_report(self, all_results):
        """生成综合报告"""
        print("\n=== 生成综合场景预测报告 ===")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Holt-Winters Gas Flow Scenario Prediction Report")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 历史数据概况
        report_lines.append("1. Historical Data Overview")
        report_lines.append("-"*30)
        report_lines.append(f"Total Records: {len(self.df)}")
        
        workday_avg = self.df[self.df['is_weekend'] == 0]['total_flow'].mean()
        weekend_avg = self.df[self.df['is_weekend'] == 1]['total_flow'].mean()
        workday_increase = (workday_avg / weekend_avg - 1) * 100 if weekend_avg > 0 else 0
        
        report_lines.append(f"Historical Weekday Average Flow: {workday_avg:.2f}")
        report_lines.append(f"Historical Weekend Average Flow: {weekend_avg:.2f}")
        report_lines.append(f"Weekday vs Weekend Increase: {workday_increase:.1f}%")
        report_lines.append("")
        
        # 各场景预测结果
        for scenario_idx, (scenario, results) in enumerate(all_results.items(), 2):
            report_lines.append(f"{scenario_idx}. {scenario} Prediction Results")
            report_lines.append("-"*30)
            
            for model_name, predictions in results.items():
                if predictions is not None:
                    analysis = self.analyze_predictions(predictions)
                    
                    report_lines.append(f"Holt-Winters Model:")
                    report_lines.append(f"  Average Flow: {analysis['avg_flow']:.2f}")
                    report_lines.append(f"  Maximum Flow: {analysis['max_flow']:.2f}")
                    report_lines.append(f"  Minimum Flow: {analysis['min_flow']:.2f}")
                    report_lines.append(f"  Morning (6-12h): {analysis['morning_avg']:.2f}")
                    report_lines.append(f"  Afternoon (12-18h): {analysis['afternoon_avg']:.2f}")
                    report_lines.append(f"  Work vs Non-work Hours: {analysis['worktime_increase']:.1f}%")
                    report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'holt_winters_scenario_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"报告已保存到 {report_path}")
        print(report_text)
    
    def run_all_scenarios(self):
        """运行所有场景测试"""
        # 定义测试场景 - 使用英文名称
        scenarios = {
            'Today_Jun19': datetime(2025, 6, 19, 0, 0),
            'Weekend_Prediction': datetime(2025, 6, 21, 0, 0),  # Saturday
            'Weekday_Prediction': datetime(2025, 6, 23, 0, 0),  # Monday
        }
        
        all_results = {}
        
        for scenario_name, start_time in scenarios.items():
            results = self.run_scenario_test(scenario_name, start_time)
            all_results[scenario_name] = results
        
        # 生成对比图表和报告
        self.create_scenario_comparison(all_results)
        self.generate_comprehensive_report(all_results)
        
        return all_results

def main():
    # 创建预测器
    predictor = AdvancedScenarioPredictor()
    
    # 加载数据
    predictor.load_data()
    
    # 训练模型
    predictor.train_all_models()
    
    # 运行所有场景测试
    results = predictor.run_all_scenarios()
    
    print("\n" + "="*60)
    print("Holt-Winters Scenario Prediction Test Completed!")
    print("="*60)
    print("Generated Files:")
    print("1. Model Files: models/")
    print("2. Prediction Results: test_predictions/")
    print("3. Visualization Chart: visualizations/")
    print("4. Report: test_predictions/holt_winters_scenario_report.txt")

if __name__ == "__main__":
    main() 
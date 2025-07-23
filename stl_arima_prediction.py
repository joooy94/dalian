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
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

class STLARIMAPredictor:
    """STL-ARIMA气体流量预测器 - 使用7天数据预测特定日期"""
    
    def __init__(self, data_file='历史数据2_total_flow.csv', 
                 model_dir='models', 
                 output_dir='stl_arima_predictions',
                 viz_dir='visualizations'):
        
        self.data_file = data_file
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.viz_dir = viz_dir
        
        # 创建目录
        for directory in [self.model_dir, self.output_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.df = None
        self.stl_model = None
        self.arima_model = None
        self.seasonal_component = None
        self.trend_component = None
        
        # 设置英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self, training_days=7):
        """加载数据并选择最近N天作为训练数据"""
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
        
        print(f"训练数据: {len(self.training_data)}条记录")
        print(f"时间范围: {self.training_data['time'].min()} 到 {self.training_data['time'].max()}")
        print(f"训练数据平均流量: {self.training_data['total_flow'].mean():.2f}")
        print(f"工作日平均流量: {self.weekday_avg:.2f}")
        print(f"周末平均流量: {self.weekend_avg:.2f}")
        print(f"周末/工作日比例: {self.weekend_ratio:.3f}")
        return self
    
    def check_stationarity(self, timeseries):
        """检查时间序列的平稳性"""
        result = adfuller(timeseries.dropna())
        adf_statistic = result[0]
        p_value = result[1]
        
        print(f"ADF统计量: {adf_statistic:.6f}")
        print(f"p值: {p_value:.6f}")
        
        if p_value <= 0.05:
            print("序列是平稳的")
            return True
        else:
            print("序列不平稳，需要差分")
            return False
    
    def auto_arima_order(self, residuals, max_p=3, max_d=2, max_q=3):
        """自动选择ARIMA参数"""
        print("正在自动选择ARIMA参数...")
        
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        temp_model = ARIMA(residuals, order=(p, d, q))
                        temp_fit = temp_model.fit()
                        if temp_fit.aic < best_aic:
                            best_aic = temp_fit.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"最佳ARIMA参数: {best_order}, AIC: {best_aic:.2f}")
        return best_order
    
    def train_stl_arima(self):
        """训练STL-ARIMA模型"""
        print("训练 STL-ARIMA 模型（基于3天数据）...")
        
        try:
            # 分别处理工作日和周末数据
            weekday_data = self.training_data[self.training_data['is_weekend'] == 0]['total_flow']
            weekend_data = self.training_data[self.training_data['is_weekend'] == 1]['total_flow']
            
            print(f"工作日数据点: {len(weekday_data)}")
            print(f"周末数据点: {len(weekend_data)}")
            
            # 如果周末数据不足，使用全部数据但应用周末修正因子
            if len(weekend_data) < 144:  # 少于一天的数据
                print("周末数据不足，使用全部数据训练基础模型...")
                flow_series = self.training_data['total_flow'].values
                self.has_separate_weekend_model = False
            else:
                print("数据充足，将为工作日和周末分别建模...")
                flow_series = self.training_data['total_flow'].values
                self.has_separate_weekend_model = True
            
            # Step 1: STL分解
            print("Step 1: 进行STL分解...")
            seasonal_period = 144  # 24小时 = 144个10分钟间隔
            
            # 如果数据不够，调整季节周期
            if len(flow_series) < seasonal_period * 2:
                seasonal_period = max(24, len(flow_series) // 3)
                print(f"调整季节周期为: {seasonal_period}")
            
            # 创建pandas Series但不指定频率，让STL自动处理
            ts_data = pd.Series(flow_series)
            
            # 执行STL分解，不依赖频率
            print("使用seasonal_decompose进行分解...")
            decomposition = seasonal_decompose(ts_data, model='additive', period=seasonal_period)
            
            # 保存分解结果
            self.seasonal_component = decomposition.seasonal
            self.trend_component = decomposition.trend
            residuals = decomposition.resid.dropna()  # 去除NaN值
            
            print(f"季节性分解完成 - 季节周期: {seasonal_period}")
            
            # Step 2: 检查残差的平稳性
            print("\nStep 2: 检查残差平稳性...")
            self.check_stationarity(residuals)
            
            # Step 3: 自动选择ARIMA参数
            arima_order = self.auto_arima_order(residuals)
            
            # Step 4: 训练ARIMA模型
            print(f"\nStep 3: 训练ARIMA{arima_order}模型...")
            self.arima_model = ARIMA(residuals, order=arima_order)
            arima_fit = self.arima_model.fit()
            
            print("ARIMA模型训练完成")
            print(f"模型摘要: AIC={arima_fit.aic:.2f}, BIC={arima_fit.bic:.2f}")
            
            # 如果有足够的周末数据，计算周末特有的季节性模式
            if self.has_separate_weekend_model and len(weekend_data) >= 144:
                print("\nStep 4: 计算周末特有的季节性模式...")
                self._calculate_weekend_seasonal_pattern()
            
            # 保存模型和组件
            self.save_stl_arima_model(arima_fit, arima_order, seasonal_period)
            
            return True
            
        except Exception as e:
            print(f"STL-ARIMA 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_stl_arima_model(self, arima_fit, arima_order, seasonal_period):
        """保存STL-ARIMA模型"""
        # 保存ARIMA模型
        arima_path = os.path.join(self.model_dir, 'stl_arima_model.pkl')
        with open(arima_path, 'wb') as f:
            pickle.dump(arima_fit, f)
        
        # 保存STL组件
        stl_components = {
            'seasonal': self.seasonal_component,
            'trend': self.trend_component,
            'seasonal_period': seasonal_period,
            'weekend_seasonal': getattr(self, 'weekend_seasonal', None),
            'has_separate_weekend_model': getattr(self, 'has_separate_weekend_model', False),
            'weekday_avg': self.weekday_avg,
            'weekend_avg': self.weekend_avg,
            'weekend_ratio': self.weekend_ratio
        }
        stl_path = os.path.join(self.model_dir, 'stl_components.pkl')
        with open(stl_path, 'wb') as f:
            pickle.dump(stl_components, f)
        
        # 保存模型信息
        model_info = f"""STL-ARIMA Model (Enhanced with Weekend/Weekday Differentiation)
Training Data: {len(self.training_data)} points ({len(self.training_data)/144:.1f} days)
ARIMA Order: {arima_order}
Seasonal Period: {seasonal_period}
Time Range: {self.training_data['time'].min()} to {self.training_data['time'].max()}
Average Flow: {self.training_data['total_flow'].mean():.2f}
Weekday Average: {self.weekday_avg:.2f}
Weekend Average: {self.weekend_avg:.2f}
Weekend/Weekday Ratio: {self.weekend_ratio:.3f}
Separate Weekend Model: {getattr(self, 'has_separate_weekend_model', False)}
AIC: {arima_fit.aic:.2f}
BIC: {arima_fit.bic:.2f}
"""
        info_path = os.path.join(self.model_dir, 'stl_arima_info.txt')
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(model_info)
        
        print(f"增强版STL-ARIMA模型已保存到 {self.model_dir}/")
        print(f"包含工作日/周末差异化处理：周末修正比例 {self.weekend_ratio:.3f}")
    
    def predict_seasonal_trend(self, target_start, steps=144, is_weekend=False):
        """预测季节性和趋势组件，根据工作日/周末使用不同模式"""
        try:
            # 根据是否为周末选择不同的季节性模式
            if is_weekend and hasattr(self, 'weekend_seasonal') and self.weekend_seasonal is not None:
                print(f"使用周末季节性模式进行预测...")
                seasonal_pattern = self.weekend_seasonal.values
            else:
                print(f"使用工作日季节性模式进行预测...")
                seasonal_pattern = self.seasonal_component.values
                # 如果是周末但没有专门的周末模式，应用比例调整
                if is_weekend:
                    seasonal_pattern = seasonal_pattern * self.weekend_ratio
                    print(f"应用周末修正因子: {self.weekend_ratio:.3f}")
            
            seasonal_period = len(seasonal_pattern) if len(seasonal_pattern) < 144 else 144
            
            # 延拓季节性组件
            seasonal_forecast = []
            for i in range(steps):
                pattern_index = i % seasonal_period
                if pattern_index < len(seasonal_pattern):
                    seasonal_forecast.append(seasonal_pattern[pattern_index])
                else:
                    # 如果超出模式长度，使用平均值
                    seasonal_forecast.append(np.mean(seasonal_pattern))
            
            # 趋势组件（使用最后几个值的平均值或线性外推）
            trend_values = self.trend_component.dropna()
            if len(trend_values) >= 10:
                # 使用最后10个值的线性趋势
                last_10_trend = trend_values.tail(10).values
                x = np.arange(len(last_10_trend))
                coeffs = np.polyfit(x, last_10_trend, 1)
                
                trend_forecast = []
                for i in range(steps):
                    trend_val = coeffs[0] * (len(last_10_trend) + i) + coeffs[1]
                    trend_forecast.append(trend_val)
            else:
                # 使用最后的趋势值
                last_trend = trend_values.iloc[-1] if len(trend_values) > 0 else 0
                trend_forecast = [last_trend] * steps
            
            # 如果是周末，对趋势也应用修正
            if is_weekend:
                trend_forecast = np.array(trend_forecast) * self.weekend_ratio
                print(f"趋势组件也应用周末修正因子")
            
            return np.array(seasonal_forecast), np.array(trend_forecast)
            
        except Exception as e:
            print(f"预测季节性和趋势组件失败: {e}")
            # 返回零值作为备选
            return np.zeros(steps), np.zeros(steps)
    
    def predict_single_day(self, target_date):
        """使用STL-ARIMA预测单个日期的24小时数据"""
        if self.arima_model is None:
            print("模型未训练，请先调用 train_stl_arima()")
            return None
        
        try:
            steps = 144  # 24小时 = 144个10分钟间隔
            target_start = datetime.combine(target_date, datetime.min.time())
            
            # 判断目标日期是否为周末
            is_weekend = target_date.weekday() >= 5
            day_type = "周末" if is_weekend else "工作日"
            
            print(f"  使用STL-ARIMA预测 {target_date} ({day_type})")
            
            # 加载保存的模型和组件
            arima_path = os.path.join(self.model_dir, 'stl_arima_model.pkl')
            with open(arima_path, 'rb') as f:
                arima_fit = pickle.load(f)
            
            # 加载STL组件和周末参数
            if not hasattr(self, 'seasonal_component') or self.seasonal_component is None:
                self.load_saved_components()
            
            # Step 1: ARIMA预测残差
            residual_forecast = arima_fit.forecast(steps=steps)
            
            # Step 2: 预测季节性和趋势组件（传入周末标识）
            seasonal_forecast, trend_forecast = self.predict_seasonal_trend(target_start, steps, is_weekend)
            
            # Step 3: 组合预测结果
            final_forecast = trend_forecast + seasonal_forecast + residual_forecast
            
            # 创建时间序列
            time_range = pd.date_range(
                start=target_start, 
                periods=steps, 
                freq='10T'
            )
            
            pred_df = pd.DataFrame({
                'time': time_range,
                'predicted_flow': final_forecast,
                'trend': trend_forecast,
                'seasonal': seasonal_forecast,
                'residual': residual_forecast
            })
            
            # 添加时间特征
            pred_df['hour'] = pred_df['time'].dt.hour
            pred_df['day_of_week'] = pred_df['time'].dt.dayofweek
            pred_df['is_weekend'] = (pred_df['day_of_week'] >= 5).astype(int)
            
            return pred_df
            
        except Exception as e:
            print(f"STL-ARIMA预测日期 {target_date} 失败: {e}")
            return None
    
    def get_historical_data_for_date(self, target_date):
        """获取指定日期的真实历史数据"""
        target_date_only = target_date.date()
        mask = (self.df['time'].dt.date == target_date_only)
        historical_data = self.df[mask].copy()
        
        if len(historical_data) > 0:
            historical_data['hour'] = historical_data['time'].dt.hour
            historical_data['day_of_week'] = historical_data['time'].dt.dayofweek
            return historical_data[['time', 'total_flow', 'hour', 'day_of_week']].copy()
        
        # 寻找相同星期几的数据
        target_weekday = target_date.weekday()
        self.df['day_of_week'] = self.df['time'].dt.dayofweek
        weekday_mask = (self.df['day_of_week'] == target_weekday)
        weekday_data = self.df[weekday_mask].copy()
        
        if len(weekday_data) > 0:
            training_start = self.training_data['time'].min()
            recent_weekday_data = weekday_data[weekday_data['time'] >= training_start]
            
            if len(recent_weekday_data) > 0:
                unique_dates = recent_weekday_data['time'].dt.date.unique()
                if len(unique_dates) > 0:
                    latest_date = max(unique_dates)
                    sample_date_data = recent_weekday_data[recent_weekday_data['time'].dt.date == latest_date].copy()
                    sample_date_data['hour'] = sample_date_data['time'].dt.hour
                    sample_date_data['day_of_week'] = sample_date_data['time'].dt.dayofweek
                    return sample_date_data[['time', 'total_flow', 'hour', 'day_of_week']].copy()
        
        return None
    
    def run_target_predictions(self):
        """预测指定的目标日期"""
        target_dates = [
            datetime(2025, 6, 11).date(),  # 6月11日
            datetime(2025, 6, 14).date(),  # 6月14日
            datetime(2025, 6, 18).date(),  # 6月18日
            datetime(2025, 6, 19).date(),  # 6月19日
        ]
        
        print("\n=== 开始STL-ARIMA预测目标日期 ===")
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
                filename = f"stl_arima_prediction_{date_str}.csv"
                filepath = os.path.join(self.output_dir, filename)
                predictions.to_csv(filepath, index=False, encoding='utf-8')
                print(f"  已保存到: {filename}")
        
        return results
    
    def create_prediction_visualization(self, all_results):
        """创建STL-ARIMA预测结果可视化"""
        print("\n=== 生成STL-ARIMA预测对比图表 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('STL-ARIMA: 7-Day Training Gas Flow Predictions vs Historical Data', fontsize=16, fontweight='bold')
        
        target_dates = list(all_results.keys())
        
        for idx, target_date in enumerate(target_dates):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            predictions = all_results[target_date]
            
            # 绘制预测数据
            ax.plot(predictions['time'], predictions['predicted_flow'], 
                   label='STL-ARIMA Prediction', color='#2ca02c', 
                   linewidth=2.5, alpha=0.9)
            
            # 绘制趋势组件
            ax.plot(predictions['time'], predictions['trend'], 
                   label='Trend Component', color='#d62728', 
                   linewidth=1.5, alpha=0.7, linestyle=':')
            
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
        save_path = os.path.join(self.viz_dir, 'stl_arima_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"STL-ARIMA可视化图表已保存到: {save_path}")
    
    def create_decomposition_plot(self):
        """创建季节性分解图"""
        if self.seasonal_component is None:
            return
            
        print("生成季节性分解图...")
        
        try:
            flow_series = self.training_data['total_flow'].values
            ts_data = pd.Series(flow_series)
            
            seasonal_period = 144
            if len(flow_series) < seasonal_period * 2:
                seasonal_period = max(24, len(flow_series) // 3)
            
            decomposition = seasonal_decompose(ts_data, model='additive', period=seasonal_period)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle('Seasonal Decomposition of Gas Flow Data', fontsize=16, fontweight='bold')
            
            # 原始数据
            axes[0].plot(range(len(ts_data)), ts_data.values, color='black', linewidth=1)
            axes[0].set_title('Original Time Series')
            axes[0].set_ylabel('Gas Flow')
            axes[0].grid(True, alpha=0.3)
            
            # 趋势
            trend_data = decomposition.trend.dropna()
            axes[1].plot(range(len(trend_data)), trend_data.values, color='red', linewidth=1.5)
            axes[1].set_title('Trend Component')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            # 季节性
            seasonal_data = decomposition.seasonal
            axes[2].plot(range(len(seasonal_data)), seasonal_data.values, color='green', linewidth=1)
            axes[2].set_title('Seasonal Component')
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            # 残差
            resid_data = decomposition.resid.dropna()
            axes[3].plot(range(len(resid_data)), resid_data.values, color='blue', linewidth=1)
            axes[3].set_title('Residual Component')
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Time Points')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            decomp_path = os.path.join(self.viz_dir, 'seasonal_decomposition.png')
            plt.savefig(decomp_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"季节性分解图已保存到: {decomp_path}")
            
        except Exception as e:
            print(f"生成季节性分解图失败: {e}")
    
    def generate_prediction_report(self, all_results):
        """生成STL-ARIMA预测报告"""
        print("\n=== 生成STL-ARIMA预测报告 ===")
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("STL-ARIMA Gas Flow Prediction Report")
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
        
        # 模型信息
        report_lines.append("2. STL-ARIMA Model Information")
        report_lines.append("-"*30)
        try:
            with open(os.path.join(self.model_dir, 'stl_arima_info.txt'), 'r', encoding='utf-8') as f:
                model_info = f.read()
                report_lines.extend(model_info.split('\n')[1:])  # 跳过第一行标题
        except:
            report_lines.append("Model information not available")
        report_lines.append("")
        
        # 各日期预测结果
        for idx, (target_date, predictions) in enumerate(all_results.items(), 3):
            date_str = target_date.strftime("%Y-%m-%d")
            weekday = target_date.strftime("%A")
            
            report_lines.append(f"{idx}. STL-ARIMA Prediction for {date_str} ({weekday})")
            report_lines.append("-"*30)
            
            avg_flow = predictions['predicted_flow'].mean()
            max_flow = predictions['predicted_flow'].max()
            min_flow = predictions['predicted_flow'].min()
            
            # 组件分析
            avg_trend = predictions['trend'].mean()
            avg_seasonal = predictions['seasonal'].mean()
            avg_residual = predictions['residual'].mean()
            
            # 时间段分析
            morning_mask = (predictions['hour'] >= 6) & (predictions['hour'] < 12)
            afternoon_mask = (predictions['hour'] >= 12) & (predictions['hour'] < 18)
            worktime_mask = (predictions['hour'] >= 8) & (predictions['hour'] <= 18)
            
            morning_avg = predictions[morning_mask]['predicted_flow'].mean() if morning_mask.any() else 0
            afternoon_avg = predictions[afternoon_mask]['predicted_flow'].mean() if afternoon_mask.any() else 0
            worktime_avg = predictions[worktime_mask]['predicted_flow'].mean() if worktime_mask.any() else 0
            non_worktime_avg = predictions[~worktime_mask]['predicted_flow'].mean() if (~worktime_mask).any() else 0
            
            worktime_diff = ((worktime_avg / non_worktime_avg - 1) * 100) if non_worktime_avg > 0 else 0
            
            report_lines.append(f"  Total Prediction:")
            report_lines.append(f"    Average Flow: {avg_flow:.2f}")
            report_lines.append(f"    Maximum Flow: {max_flow:.2f}")
            report_lines.append(f"    Minimum Flow: {min_flow:.2f}")
            report_lines.append(f"  Component Analysis:")
            report_lines.append(f"    Average Trend: {avg_trend:.2f}")
            report_lines.append(f"    Average Seasonal: {avg_seasonal:.2f}")
            report_lines.append(f"    Average Residual: {avg_residual:.2f}")
            report_lines.append(f"  Time Period Analysis:")
            report_lines.append(f"    Morning (6-12h): {morning_avg:.2f}")
            report_lines.append(f"    Afternoon (12-18h): {afternoon_avg:.2f}")
            report_lines.append(f"    Work vs Non-work Hours: {worktime_diff:.1f}%")
            
            # 与训练数据平均值的比较
            training_avg = self.training_data['total_flow'].mean()
            diff_from_training = ((avg_flow / training_avg - 1) * 100) if training_avg > 0 else 0
            report_lines.append(f"    Difference from Training Average: {diff_from_training:.1f}%")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'stl_arima_prediction_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"STL-ARIMA报告已保存到: {report_path}")
        print("\n" + report_text)
    
    def _calculate_weekend_seasonal_pattern(self):
        """计算周末特有的季节性模式"""
        try:
            # 获取周末数据
            weekend_data = self.training_data[self.training_data['is_weekend'] == 1]['total_flow']
            
            if len(weekend_data) >= 144:
                # 对周末数据进行季节性分解
                weekend_series = pd.Series(weekend_data.values)
                seasonal_period = min(144, len(weekend_data) // 2)
                
                weekend_decomp = seasonal_decompose(weekend_series, model='additive', period=seasonal_period)
                self.weekend_seasonal = weekend_decomp.seasonal
                
                print(f"周末季节性模式计算完成，周期: {seasonal_period}")
            else:
                # 数据不足时，基于工作日模式调整
                self.weekend_seasonal = self.seasonal_component * self.weekend_ratio
                print("周末数据不足，基于工作日模式调整")
                
        except Exception as e:
            print(f"计算周末季节性模式失败: {e}")
            # 使用比例调整作为备选方案
            self.weekend_seasonal = self.seasonal_component * self.weekend_ratio if self.weekend_ratio > 0 else self.seasonal_component * 0.8
    
    def load_saved_components(self):
        """加载保存的STL组件和周末参数"""
        try:
            stl_path = os.path.join(self.model_dir, 'stl_components.pkl')
            with open(stl_path, 'rb') as f:
                components = pickle.load(f)
            
            self.seasonal_component = components.get('seasonal')
            self.trend_component = components.get('trend')
            self.weekend_seasonal = components.get('weekend_seasonal')
            self.has_separate_weekend_model = components.get('has_separate_weekend_model', False)
            self.weekday_avg = components.get('weekday_avg', 0)
            self.weekend_avg = components.get('weekend_avg', 0)
            self.weekend_ratio = components.get('weekend_ratio', 0.8)
            
            print(f"已加载STL组件和周末参数，周末修正比例: {self.weekend_ratio:.3f}")
            return True
            
        except Exception as e:
            print(f"加载STL组件失败: {e}")
            return False

def main():
    # 创建STL-ARIMA预测器
    predictor = STLARIMAPredictor()
    
    # 加载7天训练数据
    predictor.load_data(training_days=7)
    
    # 训练STL-ARIMA模型
    if predictor.train_stl_arima():
        # 生成季节性分解图
        predictor.create_decomposition_plot()
        
        # 预测目标日期
        results = predictor.run_target_predictions()
        
        if results:
            # 生成可视化和报告
            predictor.create_prediction_visualization(results)
            predictor.generate_prediction_report(results)
            
            print("\n" + "="*60)
            print("STL-ARIMA预测完成！")
            print("="*60)
            print("生成的文件:")
            print(f"1. 模型文件: {predictor.model_dir}/")
            print(f"2. 预测结果: {predictor.output_dir}/")
            print(f"3. 可视化图表: {predictor.viz_dir}/stl_arima_predictions.png")
            print(f"4. 季节性分解图: {predictor.viz_dir}/seasonal_decomposition.png")
            print(f"5. 分析报告: {predictor.output_dir}/stl_arima_prediction_report.txt")
        else:
            print("预测失败，请检查数据和模型。")
    else:
        print("STL-ARIMA模型训练失败。")

if __name__ == "__main__":
    main() 
#!/opt/anaconda3/envs/water/bin/python
# -*- coding: utf-8 -*-
"""
诊断预测值偏大问题
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

def diagnose_prediction_bias():
    """诊断预测值偏大的原因"""
    
    print("🔍 诊断预测值偏大问题")
    print("=" * 50)
    
    # 1. 加载和处理数据
    print("1. 数据加载和处理分析")
    print("-" * 30)
    
    df = pd.read_excel('test.xlsx', sheet_name='Sheet2')
    df['TagTime'] = pd.to_datetime(df['TagTime'])
    df.set_index('TagTime', inplace=True)
    df.sort_index(inplace=True)
    
    raw_data = df['总系统瞬时（计算）'].copy()
    print(f"原始数据统计:")
    print(f"  数量: {len(raw_data)}")
    print(f"  范围: {raw_data.min():.2f} ~ {raw_data.max():.2f}")
    print(f"  均值: {raw_data.mean():.2f}")
    print(f"  中位数: {raw_data.median():.2f}")
    print(f"  标准差: {raw_data.std():.2f}")
    
    # 2. 检查平滑效果
    print(f"\n2. 平滑处理影响分析")
    print("-" * 30)
    
    clean_data = raw_data.dropna()
    smoothed_values = gaussian_filter1d(clean_data.values, sigma=5)
    smoothed_data = pd.Series(smoothed_values, index=clean_data.index)
    
    print(f"平滑后数据统计:")
    print(f"  范围: {smoothed_data.min():.2f} ~ {smoothed_data.max():.2f}")
    print(f"  均值: {smoothed_data.mean():.2f}")
    print(f"  中位数: {smoothed_data.median():.2f}")
    print(f"  标准差: {smoothed_data.std():.2f}")
    
    bias_from_smoothing = smoothed_data.mean() - raw_data.mean()
    print(f"平滑引起的均值偏差: {bias_from_smoothing:.2f}")
    
    # 3. 检查重采样效果
    print(f"\n3. 重采样处理影响分析")
    print("-" * 30)
    
    resampled_data = smoothed_data.resample('1min').mean()
    resampled_data = resampled_data.interpolate(method='time')
    
    print(f"重采样后数据统计:")
    print(f"  范围: {resampled_data.min():.2f} ~ {resampled_data.max():.2f}")
    print(f"  均值: {resampled_data.mean():.2f}")
    print(f"  中位数: {resampled_data.median():.2f}")
    print(f"  标准差: {resampled_data.std():.2f}")
    
    bias_from_resampling = resampled_data.mean() - smoothed_data.mean()
    print(f"重采样引起的均值偏差: {bias_from_resampling:.2f}")
    
    # 4. 检查工作日数据选择偏差
    print(f"\n4. 工作日数据选择偏差分析")
    print("-" * 30)
    
    df_patterns = pd.DataFrame({
        'value': resampled_data,
        'weekday': resampled_data.index.dayofweek,
        'hour': resampled_data.index.hour,
        'minute': resampled_data.index.minute,
        'date': resampled_data.index.date
    })
    
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    for weekday in range(7):
        weekday_data = df_patterns[df_patterns['weekday'] == weekday]
        if len(weekday_data) > 0:
            print(f"{weekday_names[weekday]}: 均值={weekday_data['value'].mean():.2f}, "
                  f"数量={len(weekday_data)}")
    
    # 5. 分析特定日期的训练数据
    print(f"\n5. 训练数据构建分析")
    print("-" * 30)
    
    # 模拟预测7月7日（周一）
    prediction_date = pd.to_datetime('2025-07-07').date()
    last_week_date = prediction_date - pd.Timedelta(days=7)  # 6月30日
    two_weeks_ago_date = prediction_date - pd.Timedelta(days=14)  # 6月23日
    
    print(f"预测日期: {prediction_date} (周一)")
    print(f"需要的历史数据: {two_weeks_ago_date}, {last_week_date}")
    
    # 获取周一数据
    monday_data = df_patterns[df_patterns['weekday'] == 0]
    daily_data = monday_data.groupby('date')
    
    available_dates = list(daily_data.groups.keys())
    print(f"可用的周一日期: {sorted(available_dates)}")
    
    # 检查所需日期的数据
    if last_week_date in available_dates and two_weeks_ago_date in available_dates:
        last_week_data = daily_data.get_group(last_week_date)
        two_weeks_ago_data = daily_data.get_group(two_weeks_ago_date)
        
        print(f"\n{two_weeks_ago_date} 数据统计:")
        print(f"  数量: {len(two_weeks_ago_data)}")
        print(f"  范围: {two_weeks_ago_data['value'].min():.2f} ~ {two_weeks_ago_data['value'].max():.2f}")
        print(f"  均值: {two_weeks_ago_data['value'].mean():.2f}")
        
        print(f"\n{last_week_date} 数据统计:")
        print(f"  数量: {len(last_week_data)}")
        print(f"  范围: {last_week_data['value'].min():.2f} ~ {last_week_data['value'].max():.2f}")
        print(f"  均值: {last_week_data['value'].mean():.2f}")
        
        # 构建训练数据
        training_data_list = []
        training_data_list.extend(two_weeks_ago_data['value'].values)
        training_data_list.extend(last_week_data['value'].values)
        
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        time_index = pd.date_range(start=start_time, periods=len(training_data_list), freq='1min')
        training_data = pd.Series(training_data_list, index=time_index)
        
        print(f"\n合并训练数据统计:")
        print(f"  数量: {len(training_data)}")
        print(f"  范围: {training_data.min():.2f} ~ {training_data.max():.2f}")
        print(f"  均值: {training_data.mean():.2f}")
        print(f"  中位数: {training_data.median():.2f}")
        print(f"  标准差: {training_data.std():.2f}")
        
        # 与总体数据比较
        overall_bias = training_data.mean() - resampled_data.mean()
        print(f"训练数据与总体数据的均值偏差: {overall_bias:.2f}")
        
        # 6. 分析模型预测偏差
        print(f"\n6. Holt-Winters模型预测分析")
        print("-" * 30)
        
        try:
            seasonal_periods = 1440  # 1天
            if len(training_data) < seasonal_periods * 2:
                seasonal_periods = min(len(training_data) // 3, 360)
                seasonal_periods = max(seasonal_periods, 60)
            
            print(f"使用季节性周期: {seasonal_periods}")
            
            # 测试不同的模型配置
            model_configs = [
                {'trend': 'add', 'seasonal': 'add', 'damped_trend': True},
                {'trend': 'add', 'seasonal': 'mul', 'damped_trend': True},
                {'trend': None, 'seasonal': 'add', 'damped_trend': False},
                {'trend': None, 'seasonal': None, 'damped_trend': False}
            ]
            
            for i, config in enumerate(model_configs):
                try:
                    print(f"\n测试配置 {i+1}: {config}")
                    
                    model = ExponentialSmoothing(
                        training_data,
                        trend=config['trend'],
                        seasonal=config['seasonal'],
                        seasonal_periods=seasonal_periods if config['seasonal'] else None,
                        damped_trend=config['damped_trend'],
                        initialization_method='estimated',
                        use_boxcox=False
                    )
                    
                    fitted_model = model.fit(optimized=True, remove_bias=False)
                    forecast = fitted_model.forecast(steps=1440)  # 预测1天
                    
                    print(f"  原始预测范围: {forecast.min():.2f} ~ {forecast.max():.2f}")
                    print(f"  原始预测均值: {forecast.mean():.2f}")
                    
                    # 分析预测偏差
                    prediction_bias = forecast.mean() - training_data.mean()
                    print(f"  预测均值与训练数据均值偏差: {prediction_bias:.2f}")
                    
                    negative_count = (forecast < 0).sum()
                    extreme_count = (forecast > training_data.max() * 2).sum()
                    print(f"  负值数量: {negative_count}, 极值数量: {extreme_count}")
                    
                    if i == 0:  # 详细分析第一个配置
                        # 分析模型组件
                        if hasattr(fitted_model, 'level'):
                            print(f"  模型水平值: {fitted_model.level[-1]:.2f}")
                        if hasattr(fitted_model, 'trend') and fitted_model.trend is not None:
                            print(f"  模型趋势值: {fitted_model.trend[-1]:.2f}")
                        
                        # 检查是否有趋势放大
                        trend_component = fitted_model.trend if hasattr(fitted_model, 'trend') else None
                        if trend_component is not None and len(trend_component) > 0:
                            recent_trend = trend_component[-100:].mean()  # 最近的趋势
                            print(f"  最近趋势均值: {recent_trend:.2f}")
                            
                            if recent_trend > 1.0:
                                print(f"  ⚠️  检测到正趋势，可能导致预测偏大")
                        
                except Exception as e:
                    print(f"  配置 {i+1} 失败: {e}")
        
        except Exception as e:
            print(f"模型分析失败: {e}")
    
    else:
        print("⚠️  所需的历史日期数据不完整，无法进行详细分析")
    
    # 7. 给出修复建议
    print(f"\n7. 修复建议")
    print("-" * 30)
    
    total_bias = bias_from_smoothing + bias_from_resampling + (overall_bias if 'overall_bias' in locals() else 0)
    print(f"累积偏差估计: {total_bias:.2f}")
    
    suggestions = [
        "1. 减少高斯平滑强度 (当前sigma=5，可尝试sigma=2或3)",
        "2. 检查重采样方法，考虑使用median而非mean",
        "3. 使用damped_trend=True来抑制趋势外推",
        "4. 添加预测后修正：将预测均值调整到历史均值附近",
        "5. 考虑使用移除趋势的季节性模型",
        "6. 增加训练数据的时间窗口以获得更稳定的基线"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

if __name__ == "__main__":
    diagnose_prediction_bias() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STL时间序列趋势分解工具
专门处理包含TagTime、总系统压力、总系统瞬时（计算）的Excel数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持 (macOS)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(excel_file='test.xlsx'):
    """
    加载Excel数据
    """
    print("正在加载数据...")
    
    # 读取Sheet2
    df = pd.read_excel(excel_file, sheet_name='Sheet2')
    print(f"成功加载数据，共 {len(df)} 行")
    print(f"数据列名: {list(df.columns)}")
    
    # 设置时间索引
    df['TagTime'] = pd.to_datetime(df['TagTime'])
    df.set_index('TagTime', inplace=True)
    
    # 按时间排序
    df.sort_index(inplace=True)
    
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return df

def calculate_seasonal_period(df):
    """
    根据数据频率自动计算季节性周期
    """
    # 计算时间间隔
    time_diff = df.index.to_series().diff().dropna()
    median_interval = time_diff.median()
    
    print(f"数据采样间隔: {median_interval}")
    
    # 30秒采样间隔，一天有2880个数据点，这太大了
    # 我们使用更小的周期来捕捉更短期的模式
    
    # 对于30秒采样间隔的数据：
    # 1小时 = 120个数据点
    # 2小时 = 240个数据点
    # 4小时 = 480个数据点
    # 6小时 = 720个数据点
    
    if median_interval == pd.Timedelta(seconds=30):
        # 使用2小时作为季节性周期
        period = 240
    elif median_interval <= pd.Timedelta(minutes=1):
        # 使用1小时作为季节性周期
        period = 60
    elif median_interval <= pd.Timedelta(hours=1):
        # 使用24小时周期
        period = 24
    else:
        # 使用7天周期
        period = 7
    
    # 确保周期合理
    period = min(period, len(df) // 10)  # 至少需要10个周期
    period = max(period, 10)  # 最小周期为10
    
    print(f"计算得到的季节性周期: {period}")
    return period

def perform_stl_decomposition(series, seasonal_period):
    """
    执行STL分解
    """
    print(f"正在进行STL分解，季节性周期: {seasonal_period}")
    
    # 处理缺失值
    series_clean = series.dropna()
    
    if len(series_clean) < seasonal_period * 2:
        print(f"警告: 数据点太少，无法进行STL分解")
        return None
    
    # 确保季节性周期是奇数（STL要求）
    if seasonal_period % 2 == 0:
        seasonal_period += 1
    
    # 创建一个简单的数值索引，避免频率推断问题
    series_values = pd.Series(series_clean.values, index=range(len(series_clean)))
    
    # 执行STL分解，明确指定period参数
    stl = STL(series_values, seasonal=seasonal_period, period=seasonal_period, robust=True)
    result = stl.fit()
    
    # 将结果映射回原始时间索引
    original_index = series_clean.index
    
    return {
        'original': series_clean,
        'trend': pd.Series(result.trend.values, index=original_index),
        'seasonal': pd.Series(result.seasonal.values, index=original_index),
        'resid': pd.Series(result.resid.values, index=original_index)
    }

def plot_decomposition(result, column_name, save_path):
    """
    绘制STL分解结果
    """
    print(f"正在绘制 {column_name} 的分解图...")
    
    # 创建图形
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'{column_name} - STL时间序列分解', fontsize=16, fontweight='bold')
    
    # 原始数据
    axes[0].plot(result['original'].index, result['original'].values, 
                color='blue', linewidth=1)
    axes[0].set_title('原始数据', fontsize=12)
    axes[0].set_ylabel('数值')
    axes[0].grid(True, alpha=0.3)
    
    # 趋势
    axes[1].plot(result['trend'].index, result['trend'].values, 
                color='red', linewidth=2)
    axes[1].set_title('趋势 (Trend)', fontsize=12)
    axes[1].set_ylabel('数值')
    axes[1].grid(True, alpha=0.3)
    
    # 季节性
    axes[2].plot(result['seasonal'].index, result['seasonal'].values, 
                color='green', linewidth=1)
    axes[2].set_title('季节性 (Seasonal)', fontsize=12)
    axes[2].set_ylabel('数值')
    axes[2].grid(True, alpha=0.3)
    
    # 残差
    axes[3].plot(result['resid'].index, result['resid'].values, 
                color='orange', linewidth=1)
    axes[3].set_title('残差 (Residual)', fontsize=12)
    axes[3].set_ylabel('数值')
    axes[3].set_xlabel('时间')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {save_path}")
    plt.close()

def plot_comparison(result1, result2, name1, name2, save_path):
    """
    绘制两个指标的对比图
    """
    print("正在绘制对比图...")
    
    # 创建对比图
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('STL分解结果对比', fontsize=16, fontweight='bold')
    
    results = [result1, result2]
    names = [name1, name2]
    colors = ['blue', 'red']
    
    for i, component in enumerate(['original', 'trend', 'seasonal']):
        for j, (result, name, color) in enumerate(zip(results, names, colors)):
            data = result[component]
            axes[i, j].plot(data.index, data.values, color=color, linewidth=1.5)
            
            component_names = {'original': '原始数据', 'trend': '趋势', 'seasonal': '季节性'}
            axes[i, j].set_title(f'{name} - {component_names[component]}', fontsize=12)
            axes[i, j].set_ylabel('数值')
            axes[i, j].grid(True, alpha=0.3)
            
            if i == 2:  # 最后一行添加x轴标签
                axes[i, j].set_xlabel('时间')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {save_path}")
    plt.close()

def save_results_to_csv(result, column_name):
    """
    保存分解结果到CSV文件
    """
    # 创建DataFrame
    df = pd.DataFrame({
        '时间': result['original'].index,
        '原始数据': result['original'].values,
        '趋势': result['trend'].values,
        '季节性': result['seasonal'].values,
        '残差': result['resid'].values
    })
    
    # 保存到CSV
    filename = f"results/{column_name}_stl_decomposition.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"结果已保存到: {filename}")

def main():
    """
    主函数
    """
    print("STL时间序列趋势分解分析")
    print("=" * 50)
    
    # 加载数据
    df = load_data()
    
    # 计算季节性周期
    seasonal_period = calculate_seasonal_period(df)
    
    # 创建结果目录
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 对两个目标列进行STL分解
    target_columns = ['总系统压力', '总系统瞬时（计算）']
    results = {}
    
    for column in target_columns:
        print(f"\n正在分析: {column}")
        
        # 执行STL分解
        result = perform_stl_decomposition(df[column], seasonal_period)
        
        if result is not None:
            results[column] = result
            
            # 绘制分解图
            save_path = f"{column}_stl_decomposition.png"
            plot_decomposition(result, column, save_path)
            
            # 保存结果到CSV
            save_results_to_csv(result, column)
    
    # 绘制对比图
    if len(results) == 2:
        columns = list(results.keys())
        plot_comparison(results[columns[0]], results[columns[1]], 
                       columns[0], columns[1], "stl_comparison.png")
    
    print("\n" + "=" * 50)
    print("STL分解分析完成！")
    print("生成的文件:")
    print("- 总系统压力_stl_decomposition.png")
    print("- 总系统瞬时（计算）_stl_decomposition.png")
    print("- stl_comparison.png")
    print("- results/总系统压力_stl_decomposition.csv")
    print("- results/总系统瞬时（计算）_stl_decomposition.csv")

if __name__ == "__main__":
    main() 
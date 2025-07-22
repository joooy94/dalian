#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势平滑工具
对STL分解得到的趋势进行多种平滑处理
适用于天然气生产数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持 (macOS)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrendSmoother:
    def __init__(self, csv_file_path):
        """
        初始化趋势平滑器
        
        Args:
            csv_file_path: STL分解结果CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.smoothed_trends = {}
        
    def load_data(self):
        """加载STL分解结果数据"""
        print(f"正在加载数据: {self.csv_file_path}")
        
        self.data = pd.read_csv(self.csv_file_path)
        self.data['时间'] = pd.to_datetime(self.data['时间'])
        
        print(f"成功加载数据，共 {len(self.data)} 行")
        print(f"数据列名: {list(self.data.columns)}")
        
        return True
    
    def moving_average_smooth(self, window_sizes=[10, 30, 60, 120]):
        """
        移动平均平滑
        
        Args:
            window_sizes: 不同的窗口大小列表
        """
        print("正在进行移动平均平滑...")
        
        trend_data = self.data['趋势'].values
        
        for window in window_sizes:
            # 确保窗口大小不超过数据长度
            actual_window = min(window, len(trend_data) // 2)
            
            # 计算移动平均
            smoothed = pd.Series(trend_data).rolling(window=actual_window, center=True).mean()
            
            # 填充边界值
            smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
            
            self.smoothed_trends[f'移动平均_{actual_window}点'] = smoothed.values
            
            print(f"完成移动平均平滑 (窗口大小: {actual_window})")
    
    def savgol_smooth(self, window_lengths=[21, 51, 101], polyorders=[2, 3]):
        """
        Savitzky-Golay滤波平滑
        
        Args:
            window_lengths: 窗口长度列表
            polyorders: 多项式阶数列表
        """
        print("正在进行Savitzky-Golay滤波平滑...")
        
        trend_data = self.data['趋势'].values
        
        for window_length in window_lengths:
            for polyorder in polyorders:
                # 确保窗口长度是奇数且不超过数据长度
                actual_window = min(window_length, len(trend_data) - 1)
                if actual_window % 2 == 0:
                    actual_window -= 1
                
                # 确保多项式阶数小于窗口长度
                actual_polyorder = min(polyorder, actual_window - 1)
                
                if actual_window >= 3 and actual_polyorder >= 1:
                    smoothed = signal.savgol_filter(trend_data, actual_window, actual_polyorder)
                    self.smoothed_trends[f'SavGol_{actual_window}_{actual_polyorder}阶'] = smoothed
                    print(f"完成SavGol平滑 (窗口: {actual_window}, 阶数: {actual_polyorder})")
    
    def gaussian_smooth(self, sigmas=[1, 2, 5, 10]):
        """
        高斯滤波平滑
        
        Args:
            sigmas: 高斯核标准差列表
        """
        print("正在进行高斯滤波平滑...")
        
        trend_data = self.data['趋势'].values
        
        for sigma in sigmas:
            smoothed = gaussian_filter1d(trend_data, sigma=sigma)
            self.smoothed_trends[f'高斯滤波_σ{sigma}'] = smoothed
            print(f"完成高斯滤波平滑 (σ={sigma})")
    
    def exponential_smooth(self, alphas=[0.1, 0.3, 0.5, 0.7]):
        """
        指数平滑
        
        Args:
            alphas: 平滑系数列表
        """
        print("正在进行指数平滑...")
        
        trend_data = self.data['趋势'].values
        
        for alpha in alphas:
            smoothed = np.zeros_like(trend_data)
            smoothed[0] = trend_data[0]
            
            for i in range(1, len(trend_data)):
                smoothed[i] = alpha * trend_data[i] + (1 - alpha) * smoothed[i-1]
            
            self.smoothed_trends[f'指数平滑_α{alpha}'] = smoothed
            print(f"完成指数平滑 (α={alpha})")
    
    def polynomial_smooth(self, degrees=[2, 3, 4, 5]):
        """
        多项式拟合平滑
        
        Args:
            degrees: 多项式度数列表
        """
        print("正在进行多项式拟合平滑...")
        
        trend_data = self.data['趋势'].values
        x = np.arange(len(trend_data))
        
        for degree in degrees:
            # 多项式拟合
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            
            model = LinearRegression()
            model.fit(x_poly, trend_data)
            
            smoothed = model.predict(x_poly)
            self.smoothed_trends[f'多项式拟合_{degree}次'] = smoothed
            print(f"完成多项式拟合平滑 ({degree}次)")
    
    def lowess_smooth(self, fractions=[0.1, 0.2, 0.3, 0.5]):
        """
        LOWESS平滑 (局部加权回归)
        
        Args:
            fractions: 局部拟合比例列表
        """
        print("正在进行LOWESS平滑...")
        
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            trend_data = self.data['趋势'].values
            x = np.arange(len(trend_data))
            
            for frac in fractions:
                smoothed_result = lowess(trend_data, x, frac=frac)
                smoothed = smoothed_result[:, 1]
                self.smoothed_trends[f'LOWESS_{frac}'] = smoothed
                print(f"完成LOWESS平滑 (比例={frac})")
                
        except ImportError:
            print("LOWESS平滑需要statsmodels库，跳过此方法")
    
    def plot_comparison(self, save_path='trend_smoothing_comparison.png'):
        """
        绘制所有平滑方法的对比图
        
        Args:
            save_path: 保存路径
        """
        print("正在绘制平滑对比图...")
        
        # 计算需要的子图数量
        num_methods = len(self.smoothed_trends)
        if num_methods == 0:
            print("没有平滑结果可以绘制")
            return
        
        # 创建子图布局
        cols = 3
        rows = (num_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # 原始趋势数据
        original_trend = self.data['趋势'].values
        time_index = self.data['时间']
        
        # 绘制每种平滑方法
        method_names = list(self.smoothed_trends.keys())
        
        for i, method_name in enumerate(method_names):
            row = i // cols
            col = i % cols
            
            ax = axes[row, col]
            
            # 绘制原始趋势
            ax.plot(time_index, original_trend, color='gray', alpha=0.7, 
                   linewidth=1, label='原始趋势')
            
            # 绘制平滑后的趋势
            smoothed_data = self.smoothed_trends[method_name]
            ax.plot(time_index, smoothed_data, color='red', linewidth=2, 
                   label=f'{method_name}')
            
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('数值')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(num_methods, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
        plt.close()
    
    def plot_selected_methods(self, selected_methods=None, save_path='selected_trend_smoothing.png'):
        """
        绘制选定的平滑方法
        
        Args:
            selected_methods: 选定的方法列表，如果为None则自动选择最佳方法
            save_path: 保存路径
        """
        if selected_methods is None:
            # 自动选择几种代表性的方法
            selected_methods = []
            if '移动平均_60点' in self.smoothed_trends:
                selected_methods.append('移动平均_60点')
            if 'SavGol_51_3阶' in self.smoothed_trends:
                selected_methods.append('SavGol_51_3阶')
            if '高斯滤波_σ5' in self.smoothed_trends:
                selected_methods.append('高斯滤波_σ5')
            if 'LOWESS_0.2' in self.smoothed_trends:
                selected_methods.append('LOWESS_0.2')
            
            # 如果没有找到预期的方法，使用前4个
            if not selected_methods:
                selected_methods = list(self.smoothed_trends.keys())[:4]
        
        print(f"正在绘制选定的平滑方法: {selected_methods}")
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 原始趋势数据
        original_trend = self.data['趋势'].values
        time_index = self.data['时间']
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, method_name in enumerate(selected_methods[:4]):
            if method_name in self.smoothed_trends:
                ax = axes[i]
                
                # 绘制原始趋势
                ax.plot(time_index, original_trend, color='gray', alpha=0.7, 
                       linewidth=1, label='原始趋势')
                
                # 绘制平滑后的趋势
                smoothed_data = self.smoothed_trends[method_name]
                ax.plot(time_index, smoothed_data, color=colors[i], linewidth=2, 
                       label=f'{method_name}')
                
                ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel('数值')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(len(selected_methods), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('天然气生产数据趋势平滑对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"选定方法对比图已保存到: {save_path}")
        plt.close()
    
    def save_smoothed_results(self, save_path='smoothed_trends.csv'):
        """
        保存所有平滑结果到CSV文件
        
        Args:
            save_path: 保存路径
        """
        print("正在保存平滑结果...")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            '时间': self.data['时间'],
            '原始趋势': self.data['趋势']
        })
        
        # 添加所有平滑结果
        for method_name, smoothed_data in self.smoothed_trends.items():
            result_df[method_name] = smoothed_data
        
        # 保存到CSV
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"平滑结果已保存到: {save_path}")
    
    def run_all_smoothing(self):
        """
        运行所有平滑方法
        """
        print("开始趋势平滑分析...")
        print("=" * 50)
        
        # 加载数据
        if not self.load_data():
            return
        
        # 执行各种平滑方法
        self.moving_average_smooth()
        self.savgol_smooth()
        self.gaussian_smooth()
        self.exponential_smooth()
        self.polynomial_smooth()
        self.lowess_smooth()
        
        # 绘制对比图
        self.plot_comparison()
        self.plot_selected_methods()
        
        # 保存结果
        self.save_smoothed_results()
        
        print("\n" + "=" * 50)
        print("趋势平滑分析完成！")
        print("生成的文件:")
        print("- trend_smoothing_comparison.png (所有方法对比)")
        print("- selected_trend_smoothing.png (选定方法对比)")
        print("- smoothed_trends.csv (平滑结果数据)")


def main():
    """主函数"""
    print("天然气生产数据趋势平滑工具")
    print("=" * 50)
    
    # 处理两个指标
    csv_files = [
        'results/总系统压力_stl_decomposition.csv',
        'results/总系统瞬时（计算）_stl_decomposition.csv'
    ]
    
    for csv_file in csv_files:
        try:
            print(f"\n正在处理: {csv_file}")
            
            # 创建平滑器
            smoother = TrendSmoother(csv_file)
            
            # 运行平滑分析
            smoother.run_all_smoothing()
            
            # 为不同指标创建不同的输出文件名
            indicator_name = csv_file.split('/')[-1].replace('_stl_decomposition.csv', '')
            
            # 重命名输出文件
            import os
            if os.path.exists('trend_smoothing_comparison.png'):
                os.rename('trend_smoothing_comparison.png', 
                         f'{indicator_name}_trend_smoothing_comparison.png')
            
            if os.path.exists('selected_trend_smoothing.png'):
                os.rename('selected_trend_smoothing.png', 
                         f'{indicator_name}_selected_trend_smoothing.png')
            
            if os.path.exists('smoothed_trends.csv'):
                os.rename('smoothed_trends.csv', 
                         f'{indicator_name}_smoothed_trends.csv')
            
        except Exception as e:
            print(f"处理 {csv_file} 时出错: {e}")
    
    print("\n" + "=" * 50)
    print("所有趋势平滑分析完成！")


if __name__ == "__main__":
    main() 
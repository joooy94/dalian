#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API数据天然气预测工具
使用API获取数据，然后使用现有的GasPredictor进行预测
用法: python api_prediction.py [预测日期]
例如: python api_prediction.py 2025-07-07
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# 导入自定义模块
from stl_decomposition.gas_prediction import GasPredictor
from stl_decomposition.api_data_fetcher import fetch_specific_days, save_to_csv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Prediction")

class APIGasPredictor:
    """使用API数据的天然气预测器"""
    
    def __init__(self):
        """初始化预测器"""
        self.raw_data = None
        self.smoothed_data = None
        self.resampled_data = None
        self.weekday_patterns = None
        
        # 创建原始GasPredictor实例，但不直接使用其load_and_process_data方法
        self.predictor = GasPredictor()
    
    def load_data_from_api(self, days_back=30):
        """从API加载历史数据"""
        logger.info(f"从API获取最近{days_back}天的历史数据...")
        
        # 计算日期范围
        today = datetime.now().date()
        date_list = []
        
        for i in range(days_back, 0, -1):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            date_list.append(date_str)
        
        logger.info(f"获取以下日期的数据: {date_list[0]} 到 {date_list[-1]}")
        
        # 从API获取数据
        df = fetch_specific_days(date_list, interval=60000)  # 1分钟间隔
        
        if df.empty:
            logger.error("无法从API获取数据")
            return False
        
        logger.info(f"成功获取 {len(df)} 条数据记录")
        
        # 重命名列以匹配GasPredictor期望的格式
        df.rename(columns={'总流量': '总系统瞬时（计算）'}, inplace=True)
        
        # 设置时间戳为索引
        df.set_index('timestamp', inplace=True)
        
        # 保存原始数据
        self.raw_data = df['总系统瞬时（计算）'].copy()
        
        # 保存为临时Excel文件，以便与原始GasPredictor兼容
        temp_excel = 'temp_api_data.xlsx'
        with pd.ExcelWriter(temp_excel) as writer:
            df.to_excel(writer, sheet_name='Sheet2')
        
        logger.info(f"数据已保存到临时文件: {temp_excel}")
        
        # 更新GasPredictor的excel_file属性
        self.predictor.excel_file = temp_excel
        
        return True
    
    def process_data(self):
        """处理数据，使用原始GasPredictor的方法"""
        logger.info("处理API数据...")
        
        # 调用原始GasPredictor的load_and_process_data方法
        self.predictor.load_and_process_data()
        
        # 复制处理后的数据
        self.smoothed_data = self.predictor.smoothed_data
        self.resampled_data = self.predictor.resampled_data
        self.weekday_patterns = self.predictor.weekday_patterns
        
        logger.info("数据处理完成")
        return True
    
    def predict(self, prediction_days=1, custom_prediction_date=None):
        """使用原始GasPredictor进行预测"""
        logger.info("开始预测...")
        
        # 调用原始GasPredictor的predict方法
        result = self.predictor.predict(
            prediction_days=prediction_days,
            custom_prediction_date=custom_prediction_date
        )
        
        return result
    
    def plot_prediction_results(self, prediction_series, save_path='api_prediction_plot.png'):
        """使用原始GasPredictor绘制预测结果"""
        # 调用原始GasPredictor的plot_prediction_results方法
        self.predictor.plot_prediction_results(prediction_series, save_path)


def main():
    """主函数"""
    print("🔮 API数据天然气预测工具")
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
            print("例如: python api_prediction.py 2025-07-07")
            sys.exit(1)
    else:
        print("📅 使用默认预测日期 (最新数据后一天)")
    
    # 创建API预测器
    api_predictor = APIGasPredictor()
    
    try:
        # 从API加载数据
        print("\n📊 从API加载历史数据...")
        days_back = 30  # 获取最近30天的数据
        if not api_predictor.load_data_from_api(days_back):
            print("❌ 无法从API获取数据，请检查API连接")
            sys.exit(1)
        
        # 处理数据
        print("\n📊 处理数据...")
        api_predictor.process_data()
        
        # 进行预测
        print(f"\n🔮 开始预测...")
        result = api_predictor.predict(
            prediction_days=1, 
            custom_prediction_date=custom_prediction_date
        )
        
        if result is not None:
            print(f"\n✅ 预测成功!")
            print(f"预测值范围: {result.min():.2f} ~ {result.max():.2f}")
            print(f"预测值均值: {result.mean():.2f}")
            print(f"预测值中位数: {result.median():.2f}")
            
            # 生成文件名
            if hasattr(result, 'prediction_info'):
                prediction_date = result.prediction_info['prediction_date']
                date_str = str(prediction_date).replace('-', '')
                csv_filename = f'api_prediction_{date_str}.csv'
                plot_filename = f'api_prediction_{date_str}.png'
                
                # 显示详细信息
                info = result.prediction_info
                print(f"\n📈 预测详情:")
                print(f"   预测日期: {info['prediction_date']} ({info['prediction_weekday_name']})")
                print(f"   使用的历史数据:")
                print(f"     - 第一部分: {info['two_weeks_data_date']}")
                print(f"     - 第二部分: {info['last_week_data_date']}")
                print(f"   预测时间段: 全天24小时 (1440个数据点)")
            else:
                csv_filename = 'api_prediction_result.csv'
                plot_filename = 'api_prediction_plot.png'
            
            # 保存结果
            result.to_csv(csv_filename, header=['predicted_value'])
            print(f"\n💾 结果已保存:")
            print(f"   CSV文件: {csv_filename}")
            
            # 绘制图片
            api_predictor.plot_prediction_results(result, plot_filename)
            print(f"   图片文件: {plot_filename}")
            
            # 显示关键时段预测值
            print(f"\n⏰ 关键时段预测值:")
            if hasattr(result, 'prediction_info'):
                for hour in [6, 12, 18, 23]:
                    hour_data = result[result.index.hour == hour]
                    if len(hour_data) > 0:
                        avg_value = hour_data.mean()
                        print(f"   {hour:02d}:00 时段均值: {avg_value:.2f}")
            
        else:
            print("\n❌ 预测失败!")
            print("可能原因:")
            print("  1. 指定日期没有足够的历史同工作日数据")
            print("  2. API返回的数据质量不佳")
            print("  3. 数据量不足")
            
            if custom_prediction_date:
                # 给出建议
                weekday = pd.to_datetime(custom_prediction_date).weekday()
                weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                weekday_name = weekday_names[weekday]
                
                print(f"\n💡 建议:")
                print(f"   检查是否有足够的{weekday_name}历史数据")
                last_week = custom_prediction_date - pd.Timedelta(days=7)
                two_weeks_ago = custom_prediction_date - pd.Timedelta(days=14)
                print(f"   需要的历史日期: {two_weeks_ago}, {last_week}")
                print(f"   尝试增加API获取的历史数据天数 (当前: {days_back}天)")
    
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        if os.path.exists('temp_api_data.xlsx'):
            try:
                os.remove('temp_api_data.xlsx')
                logger.info("已清理临时文件")
            except:
                pass

if __name__ == "__main__":
    main() 
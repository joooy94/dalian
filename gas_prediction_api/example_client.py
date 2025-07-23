#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天然气预测API客户端示例
演示如何使用API接口
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

class GasPredictionClient:
    def __init__(self, base_url='http://127.0.0.1:58888'):
        """初始化客户端"""
        self.base_url = base_url
        
    def check_health(self):
        """检查服务健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ 健康检查失败: {e}")
            return None
    
    def get_service_info(self):
        """获取服务信息"""
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取服务信息失败: {e}")
            return None
    
    def predict_flow(self, date):
        """预测流量"""
        try:
            url = f"{self.base_url}/predict?date={date}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ 流量预测失败: {e}")
            return None
    
    def predict_pressure(self, date):
        """预测压力"""
        try:
            url = f"{self.base_url}/predict_pressure?date={date}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ 压力预测失败: {e}")
            return None
    
    def test_prediction(self, date, metric='flow'):
        """测试预测并获取图表"""
        try:
            url = f"{self.base_url}/test?date={date}&metric={metric}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ 测试预测失败: {e}")
            return None

def demo_basic_usage():
    """基本使用示例"""
    print("🚀 天然气预测API客户端示例")
    print("=" * 50)
    
    # 创建客户端
    client = GasPredictionClient()
    
    # 1. 检查服务健康状态
    print("1. 检查服务健康状态...")
    health = client.check_health()
    if health:
        print(f"   ✅ 服务状态: {health['status']}")
        print(f"   📍 服务名称: {health['service']}")
        print(f"   📝 版本: {health['version']}")
    else:
        print("   ❌ 服务不可用，请确保API服务器正在运行")
        return
    
    print()
    
    # 2. 获取服务信息
    print("2. 获取服务信息...")
    info = client.get_service_info()
    if info:
        print(f"   📋 服务: {info['service']}")
        print(f"   📖 描述: {info['description']}")
        print(f"   🔗 可用接口:")
        for name, endpoint in info['endpoints'].items():
            print(f"      - {name}: {endpoint}")
    
    print()
    
    # 3. 预测未来日期
    future_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"3. 预测未来日期 ({future_date})...")
    
    # 流量预测
    print("   📈 预测流量...")
    flow_data = client.predict_flow(future_date)
    if flow_data and flow_data.get('success'):
        predictions = flow_data['predictions']
        df_flow = pd.DataFrame(predictions)
        df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'])
        
        print(f"   ✅ 流量预测成功!")
        print(f"      预测日期: {flow_data['prediction_date']}")
        print(f"      数据点数: {flow_data['data_points']}")
        print(f"      预测范围: {df_flow['forecast'].min():.2f} ~ {df_flow['forecast'].max():.2f}")
        print(f"      平均值: {df_flow['forecast'].mean():.2f}")
        
        # 显示前5个和后5个预测点
        print("      前5个预测点:")
        for i, row in df_flow.head().iterrows():
            print(f"        {row['timestamp'].strftime('%H:%M')} -> {row['forecast']:.2f}")
    else:
        print("   ❌ 流量预测失败")
    
    print()
    
    # 压力预测
    print("   📊 预测压力...")
    pressure_data = client.predict_pressure(future_date)
    if pressure_data and pressure_data.get('success'):
        predictions = pressure_data['predictions']
        df_pressure = pd.DataFrame(predictions)
        df_pressure['timestamp'] = pd.to_datetime(df_pressure['timestamp'])
        
        print(f"   ✅ 压力预测成功!")
        print(f"      预测日期: {pressure_data['prediction_date']}")
        print(f"      数据点数: {pressure_data['data_points']}")
        print(f"      预测范围: {df_pressure['forecast'].min():.2f} ~ {df_pressure['forecast'].max():.2f}")
        print(f"      平均值: {df_pressure['forecast'].mean():.2f}")
    else:
        print("   ❌ 压力预测失败")
    
    print()
    
    # 4. 测试预测并生成图表
    print("4. 测试预测并生成可视化图表...")
    test_result = client.test_prediction(future_date, 'flow')
    if test_result and test_result.get('success'):
        print(f"   ✅ 测试预测成功!")
        print(f"      图表文件: {test_result['plot_file']}")
        print(f"      访问地址: http://127.0.0.1:58888/static/{test_result['plot_file']}")
        print(f"      说明: {test_result['message']}")
    else:
        print("   ❌ 测试预测失败")

def demo_data_analysis():
    """数据分析示例"""
    print("\n" + "=" * 50)
    print("📊 数据分析示例")
    print("=" * 50)
    
    client = GasPredictionClient()
    future_date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    # 获取预测数据
    flow_data = client.predict_flow(future_date)
    pressure_data = client.predict_pressure(future_date)
    
    if flow_data and pressure_data:
        # 转换为DataFrame
        df_flow = pd.DataFrame(flow_data['predictions'])
        df_pressure = pd.DataFrame(pressure_data['predictions'])
        
        df_flow['timestamp'] = pd.to_datetime(df_flow['timestamp'])
        df_pressure['timestamp'] = pd.to_datetime(df_pressure['timestamp'])
        
        # 按小时分组分析
        df_flow['hour'] = df_flow['timestamp'].dt.hour
        df_pressure['hour'] = df_pressure['timestamp'].dt.hour
        
        hourly_flow = df_flow.groupby('hour')['forecast'].agg(['mean', 'min', 'max'])
        hourly_pressure = df_pressure.groupby('hour')['forecast'].agg(['mean', 'min', 'max'])
        
        print("📈 流量预测分析 (按小时):")
        print("   小时   平均值    最小值    最大值")
        print("   " + "-" * 35)
        for hour in range(0, 24, 4):  # 每4小时显示一次
            if hour in hourly_flow.index:
                row = hourly_flow.loc[hour]
                print(f"   {hour:02d}:00  {row['mean']:7.2f}  {row['min']:7.2f}  {row['max']:7.2f}")
        
        print("\n📊 压力预测分析 (按小时):")
        print("   小时   平均值    最小值    最大值")
        print("   " + "-" * 35)
        for hour in range(0, 24, 4):
            if hour in hourly_pressure.index:
                row = hourly_pressure.loc[hour]
                print(f"   {hour:02d}:00  {row['mean']:7.2f}  {row['min']:7.2f}  {row['max']:7.2f}")
        
        # 找出峰值时间
        peak_flow_hour = hourly_flow['mean'].idxmax()
        peak_pressure_hour = hourly_pressure['mean'].idxmax()
        
        print(f"\n📍 关键时间点:")
        print(f"   流量峰值时间: {peak_flow_hour:02d}:00 (平均 {hourly_flow.loc[peak_flow_hour, 'mean']:.2f})")
        print(f"   压力峰值时间: {peak_pressure_hour:02d}:00 (平均 {hourly_pressure.loc[peak_pressure_hour, 'mean']:.2f})")

def demo_error_handling():
    """错误处理示例"""
    print("\n" + "=" * 50)
    print("🛠️ 错误处理示例")
    print("=" * 50)
    
    client = GasPredictionClient()
    
    # 测试无效日期格式
    print("1. 测试无效日期格式...")
    invalid_date_result = client.predict_flow("invalid-date")
    if invalid_date_result is None:
        print("   ✅ 正确处理了无效日期格式")
    
    # 测试过去的日期（可能没有足够的历史数据）
    old_date = "2020-01-01"
    print(f"2. 测试历史日期 ({old_date})...")
    old_date_result = client.predict_flow(old_date)
    if old_date_result is None:
        print("   ✅ 正确处理了历史日期限制")
    elif not old_date_result.get('success'):
        print("   ✅ API返回了错误信息")
    
    # 测试服务连接错误
    print("3. 测试错误的服务地址...")
    wrong_client = GasPredictionClient('http://127.0.0.1:99999')  # 错误端口
    wrong_result = wrong_client.check_health()
    if wrong_result is None:
        print("   ✅ 正确处理了连接错误")

if __name__ == '__main__':
    try:
        # 基本使用示例
        demo_basic_usage()
        
        # 数据分析示例
        demo_data_analysis()
        
        # 错误处理示例
        demo_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ 示例演示完成!")
        print("📖 查看代码了解更多用法详情")
        print("🔗 API文档: http://127.0.0.1:58888/docs")
        
    except KeyboardInterrupt:
        print("\n👋 演示已停止")
    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc() 
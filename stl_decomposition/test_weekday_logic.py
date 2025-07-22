#!/opt/anaconda3/envs/water/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime, timedelta

def test_weekday_logic():
    """测试工作日逻辑"""
    
    # 模拟场景：当前最新数据是7月6日，要预测7月7日
    last_date = pd.to_datetime('2025-07-06').date()
    prediction_date = last_date + timedelta(days=1)
    
    print("=== 工作日逻辑测试 ===")
    print(f"最新数据日期: {last_date} ({pd.to_datetime(last_date).strftime('%A')})")
    print(f"预测目标日期: {prediction_date} ({pd.to_datetime(prediction_date).strftime('%A')})")
    
    # 获取预测日期的星期几
    prediction_weekday = pd.to_datetime(prediction_date).weekday()
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    prediction_weekday_name = weekday_names[prediction_weekday]
    
    print(f"预测日期是: {prediction_weekday_name}")
    
    # 应该使用的历史数据日期
    print("\n=== 应该使用的历史数据 ===")
    
    # 上周同一天 (7天前)
    last_week_date = prediction_date - timedelta(days=7)
    print(f"上周同一天: {last_week_date} ({pd.to_datetime(last_week_date).strftime('%A')})")
    
    # 上上周同一天 (14天前)  
    two_weeks_ago_date = prediction_date - timedelta(days=14)
    print(f"上上周同一天: {two_weeks_ago_date} ({pd.to_datetime(two_weeks_ago_date).strftime('%A')})")
    
    # 验证是否都是同一天
    assert pd.to_datetime(prediction_date).weekday() == pd.to_datetime(last_week_date).weekday()
    assert pd.to_datetime(prediction_date).weekday() == pd.to_datetime(two_weeks_ago_date).weekday()
    
    print("\n✅ 工作日逻辑正确！")

if __name__ == "__main__":
    test_weekday_logic() 
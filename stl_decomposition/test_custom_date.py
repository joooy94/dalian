#!/opt/anaconda3/envs/water/bin/python
# -*- coding: utf-8 -*-

from gas_prediction import GasPredictor
import pandas as pd

def test_custom_prediction():
    """测试自定义预测日期功能"""
    
    print("🧪 测试自定义预测日期功能")
    print("=" * 50)
    
    # 创建预测器
    predictor = GasPredictor('test.xlsx')
    
    try:
        # 加载数据
        predictor.load_and_process_data()
        
        # 测试不同的预测日期
        test_dates = [
            '2025-07-07',  # 周一
            '2025-07-08',  # 周二  
            '2025-07-09',  # 周三
            '2025-06-30',  # 另一个周一
        ]
        
        for test_date in test_dates:
            print(f"\n🎯 测试预测日期: {test_date}")
            print("-" * 30)
            
            result = predictor.predict(
                prediction_days=1, 
                custom_prediction_date=test_date
            )
            
            if result is not None:
                print(f"✅ 预测成功!")
                print(f"   预测值范围: {result.min():.2f} ~ {result.max():.2f}")
                
                if hasattr(result, 'prediction_info'):
                    info = result.prediction_info
                    print(f"   使用的历史数据: {info['two_weeks_data_date']}, {info['last_week_data_date']}")
                else:
                    print("   ⚠️  缺少预测信息")
            else:
                print(f"❌ 预测失败 - 可能缺少对应的历史数据")
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_prediction() 
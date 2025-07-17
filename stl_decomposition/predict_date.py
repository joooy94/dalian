#!/opt/anaconda3/envs/water/bin/python
# -*- coding: utf-8 -*-
"""
命令行天然气预测工具
用法: python predict_date.py [预测日期]
例如: python predict_date.py 2025-07-07
"""

import sys
import pandas as pd
from gas_prediction import GasPredictor

def main():
    print("🔮 天然气预测工具 - 命令行版本")
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
            print("例如: python predict_date.py 2025-07-07")
            sys.exit(1)
    else:
        print("📅 使用默认预测日期 (最新数据后一天)")
    
    # 创建预测器
    predictor = GasPredictor('test.xlsx')
    
    try:
        # 处理数据
        print("\n📊 加载和处理数据...")
        predictor.load_and_process_data()
        
        # 进行预测
        print(f"\n🔮 开始预测...")
        result = predictor.predict(
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
                csv_filename = f'prediction_{date_str}.csv'
                plot_filename = f'prediction_{date_str}.png'
                
                # 显示详细信息
                info = result.prediction_info
                print(f"\n📈 预测详情:")
                print(f"   预测日期: {info['prediction_date']} ({info['prediction_weekday_name']})")
                print(f"   使用的历史数据:")
                print(f"     - 第一部分: {info['two_weeks_data_date']}")
                print(f"     - 第二部分: {info['last_week_data_date']}")
                print(f"   预测时间段: 全天24小时 (1440个数据点)")
            else:
                csv_filename = 'prediction_result.csv'
                plot_filename = 'prediction_plot.png'
            
            # 保存结果
            result.to_csv(csv_filename, header=['predicted_value'])
            print(f"\n💾 结果已保存:")
            print(f"   CSV文件: {csv_filename}")
            
            # 绘制图片
            predictor.plot_prediction_results(result, plot_filename)
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
            print("  2. 历史数据质量不佳")
            print("  3. 数据文件 'test.xlsx' 不存在或格式错误")
            
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
    
    except FileNotFoundError:
        print("❌ 错误: 找不到数据文件 'test.xlsx'")
        print("请确保数据文件存在于当前目录")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
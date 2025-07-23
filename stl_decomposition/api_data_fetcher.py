import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("API_Data_Fetcher")

def fetch_data(start_time, end_time, interval=60000, names=None):
    """
    从API获取指定时间范围的数据
    
    参数:
        start_time (str): 开始时间，格式为 "YYYY-MM-DDTHH:MM:SS"
        end_time (str): 结束时间，格式为 "YYYY-MM-DDTHH:MM:SS"
        interval (int): 时间间隔，单位为毫秒，默认为60000（1分钟）
        names (list): 点位名称列表，默认为None
    
    返回:
        DataFrame: 包含时间戳、总流量和压力的数据框
    """
    if names is None:
        names = [
            'DLDZ_AVS_LLJ01_FI01.PV', #流量1
            'DLDZ_DQ200_LLJ01_FI01.PV', #流量2
            'DLDZ_SUM_SYSTEM_PI01.PV' #压力
        ]
    
    # 构建API请求参数
    params = {
        "startTime": start_time,
        "endTime": end_time,
        "interval": interval,
        "names": ','.join(names) if isinstance(names, list) else names
    }
    
    # API请求URL
    url = "http://8.130.25.118:8000/api/hisdata"
    
    try:
        logger.info(f"请求数据范围: {start_time} 至 {end_time}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        logger.info("API请求成功，开始处理数据...")
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        return pd.DataFrame()
    
    # 解析API响应
    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        return pd.DataFrame()
    
    # 检查API返回的数据结构
    if 'code' not in data or data['code'] != 0 or 'items' not in data or not data['items']:
        logger.warning("API返回的数据结构不符合预期或为空")
        return pd.DataFrame()
    
    # 处理数据 - 根据示例的返回格式
    all_times = set()
    data_dict = {}
    
    # 首先收集所有时间点
    for item in data['items']:
        name = item['name']
        for val_item in item['vals']:
            time_str = val_item['time']
            all_times.add(time_str)
            
            if time_str not in data_dict:
                data_dict[time_str] = {}
            
            data_dict[time_str][name] = val_item['val']
    
    # 如果没有数据，返回空DataFrame
    if not all_times:
        logger.warning("API返回数据中没有时间点")
        return pd.DataFrame()
    
    # 构建DataFrame
    rows = []
    for time_str in sorted(all_times):
        row = {'timestamp': time_str}
        row.update(data_dict[time_str])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算总流量和压力
    try:
        # 确保所需的列存在
        required_cols = ['DLDZ_AVS_LLJ01_FI01.PV', 'DLDZ_DQ200_LLJ01_FI01.PV', 'DLDZ_SUM_SYSTEM_PI01.PV']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"缺少必要的列: {missing_cols}")
            # 尝试继续处理可用的列
            for col in missing_cols:
                df[col] = None
        
        # 计算总流量和获取压力
        df['总流量'] = df['DLDZ_AVS_LLJ01_FI01.PV'].fillna(0) + df['DLDZ_DQ200_LLJ01_FI01.PV'].fillna(0)
        df['压力'] = df['DLDZ_SUM_SYSTEM_PI01.PV']
        
        # 只保留需要的列
        result_df = df[['timestamp', '总流量', '压力']].copy()
        
        # 记录数据范围
        min_time = result_df['timestamp'].min().strftime("%Y-%m-%d %H:%M")
        max_time = result_df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
        logger.info(f"数据时间范围: {min_time} 至 {max_time}, 数据点数量: {len(result_df)}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
        return pd.DataFrame()


def fetch_multiple_time_ranges(dates, interval=60000, names=None):
    """
    获取多个指定日期的全天数据并合并
    
    参数:
        dates (list): 日期列表，每个元素是一个字符串，格式为 "YYYY-MM-DD"
        interval (int): 时间间隔，单位为毫秒
        names (list): 点位名称列表
    
    返回:
        DataFrame: 合并后的数据框
    """
    all_data = []
    
    for date_str in dates:
        try:
            # 解析日期字符串
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # 设置当天的开始时间和第二天的00:00作为结束时间
            start_time = f"{date_str}T00:00:00"
            
            # 计算下一天的日期
            next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            end_time = f"{next_day}T00:00:00"
            
            logger.info(f"开始获取日期 {date_str} 的全天数据，时间范围: {start_time} 至 {end_time}")
            df = fetch_data(start_time, end_time, interval, names)
            
            if not df.empty:
                # 添加日期标识列，方便后续分析
                df['date'] = date_str
                all_data.append(df)
                logger.info(f"成功获取到 {date_str} 的 {len(df)} 条数据")
            else:
                logger.warning(f"日期 {date_str} 未获取到数据")
        
        except ValueError as e:
            logger.error(f"日期格式错误: {date_str}, 应为 YYYY-MM-DD 格式. 错误: {str(e)}")
    
    if not all_data:
        logger.warning("所有日期都未获取到数据")
        return pd.DataFrame()
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 提取日期、小时和分钟，用于排序
    merged_df['day'] = merged_df['date']
    merged_df['hour'] = merged_df['timestamp'].dt.hour
    merged_df['minute'] = merged_df['timestamp'].dt.minute
    
    # 按日期、小时、分钟排序
    logger.info("按日期、小时、分钟排序数据...")
    result_df = merged_df.sort_values(['day', 'hour', 'minute'])
    
    # 移除辅助列
    if 'hour' in result_df.columns and 'minute' in result_df.columns and 'day' in result_df.columns:
        result_df = result_df.drop(['hour', 'minute', 'day'], axis=1)
    
    logger.info(f"所有日期合并后共 {len(result_df)} 条数据")
    return result_df


def fetch_specific_days(days_list, interval=60000, names=None):
    """
    获取指定日期列表的数据
    
    参数:
        days_list (list): 日期列表，格式为 ["2025-06-01", "2025-06-08"]
        interval (int): 时间间隔，单位为毫秒
        names (list): 点位名称列表
    
    返回:
        DataFrame: 包含所有指定日期数据的数据框
    """
    return fetch_multiple_time_ranges(days_list, interval, names)


def save_to_csv(df, filename=None):
    """
    将数据保存为CSV文件
    
    参数:
        df (DataFrame): 要保存的数据框
        filename (str): 文件名，默认为当前时间
    """
    if df.empty:
        logger.warning("没有数据可保存")
        return
    
    if filename is None:
        filename = f"api_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df.to_csv(filename, index=False)
    logger.info(f"数据已保存到 {filename}")


def test_with_sample_data():
    """
    使用示例数据测试解析逻辑
    """
    sample_data = {
        "code": 0,
        "items": [
            {
                "name": "DLDZ_AVS_LLJ01_FI01.PV",
                "vals": [
                    {
                        "time": "2025-07-16T00:00:00.000",
                        "val": 46.318
                    }
                ]
            },
            {
                "name": "DLDZ_DQ200_LLJ01_FI01.PV",
                "vals": [
                    {
                        "time": "2025-07-16T00:00:00.000",
                        "val": 178.38
                    }
                ]
            },
            {
                "name": "DLDZ_SUM_SYSTEM_PI01.PV",
                "vals": [
                    {
                        "time": "2025-07-16T00:00:00.000",
                        "val": 6.0508
                    }
                ]
            }
        ]
    }
    
    logger.info("使用示例数据测试解析逻辑")
    
    # 处理示例数据
    all_times = set()
    data_dict = {}
    
    for item in sample_data['items']:
        name = item['name']
        for val_item in item['vals']:
            time_str = val_item['time']
            all_times.add(time_str)
            
            if time_str not in data_dict:
                data_dict[time_str] = {}
            
            data_dict[time_str][name] = val_item['val']
    
    rows = []
    for time_str in sorted(all_times):
        row = {'timestamp': time_str}
        row.update(data_dict[time_str])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 计算总流量和压力
    df['总流量'] = df['DLDZ_AVS_LLJ01_FI01.PV'] + df['DLDZ_DQ200_LLJ01_FI01.PV']
    df['压力'] = df['DLDZ_SUM_SYSTEM_PI01.PV']
    
    result_df = df[['timestamp', '总流量', '压力']].copy()
    
    logger.info(f"示例数据解析结果:\n{result_df}")
    return result_df


if __name__ == "__main__":
    # 示例：获取多个特定日期的数据
    specific_days = [
        "2025-07-01",  
        "2025-07-08"   
    ]
    
    df_multiple = fetch_specific_days(specific_days)
    print(df_multiple)
    if not df_multiple.empty:
        save_to_csv(df_multiple, "multiple_days_data.csv") 
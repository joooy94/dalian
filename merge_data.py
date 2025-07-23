import pandas as pd
df = pd.read_excel('/Users/wangzhuoyang/dalian/历史数据 (2).xlsx')
df['total_flow'] = (df['DLDZ_AVS_LLJ01_FI01.PV'] + df['DLDZ_DQ200_LLJ01_FI01.PV'])/2
df.drop(columns=['DLDZ_AVS_LLJ01_FI01.PV', 'DLDZ_DQ200_LLJ01_FI01.PV'], inplace=True)
df.rename(columns={'时间': 'time'}, inplace=True)
df.to_csv('/Users/wangzhuoyang/dalian/历史数据3_total_flow.csv', index=False)

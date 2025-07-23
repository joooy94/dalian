#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天然气预测API服务启动脚本
简化版启动入口
"""

import uvicorn
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_gas_prediction_server import app

if __name__ == '__main__':
    print("🚀 启动天然气预测API服务器")
    print("=" * 50)
    print("服务地址: http://127.0.0.1:58888")
    print("API文档: http://127.0.0.1:58888/docs")
    print("健康检查: http://127.0.0.1:58888/health")
    print("=" * 50)
    
    try:
        uvicorn.run(
            app, 
            host='127.0.0.1', 
            port=58888,
            log_level='info'
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 服务启动失败: {e}")
        sys.exit(1) 
#!/usr/bin/env python3
"""
测试数据库功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_database_manager

def test_database():
    """测试数据库功能"""
    try:
        print("🔍 测试数据库功能...")
        
        # 获取数据库管理器
        db_manager = get_database_manager()
        print(f"✅ 数据库管理器创建成功")
        print(f"📁 数据库路径: {db_manager.db_path}")
        
        # 检查数据库文件是否存在
        if os.path.exists(db_manager.db_path):
            print(f"✅ 数据库文件存在: {db_manager.db_path}")
            file_size = os.path.getsize(db_manager.db_path)
            print(f"📊 数据库文件大小: {file_size} 字节")
        else:
            print(f"❌ 数据库文件不存在: {db_manager.db_path}")
            
        # 测试保存历史记录
        print("\n📝 测试保存历史记录...")
        history_id = db_manager.save_analysis_history(
            asset="BTC",
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-01-31",
            trading_strategy="high_frequency",
            status="completed",
            result_summary="测试分析结果摘要",
            result_details={"analysis": "详细分析结果", "indicators": ["RSI", "MACD"]},
            session_id="test_session_123"
        )
        print(f"✅ 历史记录保存成功，ID: {history_id}")
        
        # 测试读取历史记录
        print("\n📖 测试读取历史记录...")
        history_list = db_manager.get_analysis_history_list(limit=10)
        print(f"✅ 历史记录读取成功，记录数: {len(history_list)}")
        
        if history_list:
            for i, record in enumerate(history_list):
                print(f"  {i+1}. ID: {record['id']}, 资产: {record['asset']}, 状态: {record['status']}")
        else:
            print("  📭 没有找到历史记录")
            
        # 测试根据ID读取
        if history_list:
            print("\n🔍 测试根据ID读取历史记录...")
            record = db_manager.get_analysis_history_by_id(history_id)
            if record:
                print(f"✅ 根据ID读取成功")
                print(f"   资产: {record['asset']}")
                print(f"   时间周期: {record['timeframe']}")
                print(f"   结果摘要: {record.get('result_summary', 'N/A')}")
            else:
                print("❌ 根据ID读取失败")
        
        print("\n🎉 数据库测试完成")
        
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database()
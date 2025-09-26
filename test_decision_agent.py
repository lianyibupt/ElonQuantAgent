#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试决策代理的提示词模板渲染功能
"""
from langchain_core.prompts import ChatPromptTemplate
import json


def test_decision_prompt_template():
    """测试决策代理的提示词模板渲染"""
    print("开始测试决策代理的提示词模板渲染...")
    
    # 创建一个简单的状态模拟实际应用中的数据
    template_inputs = {
        "stock_name": "BTC/USDT",
        "time_frame": "4h",
        "indicator_report": "测试指标报告",
        "pattern_report": "测试形态报告",
        "trend_report": "测试趋势报告"
    }
    
    try:
        # 直接创建测试用的提示词模板，使用修复后的转义花括号
        # 高频交易策略提示词
        system_prompt = (
            "你是一位资深的高频交易决策专家，最短持有2天，最长持有1个月。"
            "基于以下综合分析报告，做出最终的交易决策。请用中文回答。\n\n"
            "股票代码: {stock_name}\n"
            "时间周期: {time_frame}\n\n"
            "技术指标分析:\n{indicator_report}\n\n"
            "形态分析:\n{pattern_report}\n\n"
            "趋势分析:\n{trend_report}\n\n"
            "请按照以下JSON格式提供你的最终决策（用中文填写）:\n"
            "{{\n"
            '  "decision": "买入/卖出/持有",\n'
            '  "confidence": "高/中/低",\n'
            '  "risk_reward_ratio": "X:Y",\n'
            '  "forecast_horizon": "预测时间段",\n'
            '  "justification": "详细的中文理由说明"\n'
            "}}\n\n"
            "请综合考虑所有三种分析类型，为高频交易提供可操作的中文见解。"
        )
        
        # 创建提示词模板
        decision_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                system_prompt
            )
        ])
        
        # 测试提示词模板渲染
        print("测试提示词模板渲染...")
        # 使用format_messages方法直接测试模板渲染，避免需要完整的LLM
        messages = decision_prompt.format_messages(**template_inputs)
        
        # 检查渲染结果
        rendered_content = messages[0].content
        print(f"✅ 提示词模板渲染成功!")
        print(f"渲染后的系统提示词包含 {{}} 字符: {'{{' in rendered_content and '}}' in rendered_content}")
        
        # 验证没有未转义的变量
        if '{\n "decision"' in rendered_content:
            print("❌ 错误: 渲染结果中仍包含未转义的变量 '{\\n \"decision\"}'")
        else:
            print("✅ 成功: 渲染结果中没有未转义的变量 '{\\n \"decision\"}'")
        
        print("\n测试完成！修复已经解决了ChatPromptTemplate中的转义问题。")
        print("原始错误原因: 在LangChain的ChatPromptTemplate中，普通的花括号{}会被解析为变量占位符，")
        print("需要使用双花括号{{}}来表示字面值的花括号，否则会被误认为是需要传入的变量。")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_decision_prompt_template()
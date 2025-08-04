#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu英文字体修复测试
专门测试Ubuntu系统中英文字符显示问题的修复效果
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from unittest.mock import patch

# 添加项目路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_ubuntu_english_font_fix():
    """测试Ubuntu英文字体修复效果"""
    print("=" * 60)
    print("Ubuntu英文字体修复测试")
    print("=" * 60)
    
    # 模拟Ubuntu环境并导入修复后的字体设置
    with patch('platform.system') as mock_system:
        mock_system.return_value = 'Linux'
        
        # 重新导入字体设置函数
        from atscui.utils.comparative_analysis import setup_chinese_fonts
        
        print("\n正在模拟Ubuntu环境并配置字体...")
        selected_font = setup_chinese_fonts()
        
        # 获取当前字体配置
        current_fonts = plt.rcParams['font.sans-serif']
        print(f"\n当前字体配置:")
        print(f"  主要字体: {selected_font}")
        print(f"  字体优先级: {current_fonts[:5]}")
        print(f"  英文字体优先: {'Liberation Sans' in current_fonts[:3] or 'DejaVu Sans' in current_fonts[:3]}")
        print(f"  负号处理: {not plt.rcParams['axes.unicode_minus']}")
    
    # 创建英文字符重点测试图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ubuntu English Font Fix Test - 英文字体修复测试', fontsize=16, fontweight='bold')
    
    # 测试1: 纯英文标签和数值
    english_categories = ['Waiting Time', 'Speed', 'Throughput', 'Fuel Consumption', 'CO2 Emission']
    values = [25.3, -12.8, 45.7, -8.5, 19.2]
    colors = ['#E74C3C' if v > 0 else '#27AE60' for v in values]
    
    bars = ax1.bar(english_categories, values, color=colors, alpha=0.8)
    ax1.set_title('Performance Metrics (Pure English)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement Rate (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加英文数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 测试2: 英文算法名称对比
    algorithms = ['PPO Algorithm', 'DQN Method', 'A2C Approach', 'SAC Technique', 'Fixed Timing']
    scores = [85.6, 78.3, 82.1, 79.8, 65.4]
    
    ax2.plot(algorithms, scores, marker='o', linewidth=2, markersize=8, color='#3498DB')
    ax2.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Performance Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加英文数值标签
    for i, (alg, score) in enumerate(zip(algorithms, scores)):
        ax2.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 测试3: 英文单位和符号
    metrics = ['Speed (km/h)', 'Time (seconds)', 'Flow (veh/h)', 'Distance (meters)', 'Efficiency (%)']
    metric_values = [45.2, 123.7, 1850, 2340, 87.9]
    
    ax3.barh(metrics, metric_values, color='#9B59B6', alpha=0.7)
    ax3.set_title('Traffic Metrics with English Units', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Measurement Values', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 添加英文数值标签
    for i, value in enumerate(metric_values):
        ax3.text(value + 20, i, f'{value:.1f}', ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    # 测试4: 英文时间序列标签
    time_labels = ['Morning Peak\n(7-9 AM)', 'Off-Peak\n(9-17 PM)', 'Evening Peak\n(17-19 PM)', 'Night\n(19-7 AM)']
    traffic_data = [1200, 800, 1350, 400]
    
    ax4.plot(time_labels, traffic_data, marker='s', linewidth=3, markersize=10, 
             color='#E67E22', markerfacecolor='#C0392B')
    ax4.set_title('Daily Traffic Pattern (English Labels)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Traffic Volume (vehicles/hour)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 添加英文数值标签
    for i, (time, volume) in enumerate(zip(time_labels, traffic_data)):
        ax4.text(i, volume + 50, f'{volume} veh/h', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存测试图片
    output_path = 'ubuntu_english_font_fix_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Ubuntu英文字体测试图表已保存: {output_path}")
    
    return output_path

def test_mixed_language_display():
    """测试中英文混合显示效果"""
    print("\n" + "=" * 60)
    print("中英文混合显示测试")
    print("=" * 60)
    
    # 创建混合语言测试图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Mixed Language Display Test - 中英文混合显示测试', fontsize=16, fontweight='bold')
    
    # 测试1: 中英文混合标签
    mixed_labels = ['等待时间\nWaiting Time', '通行速度\nTraffic Speed', '燃油消耗\nFuel Usage', 'CO₂排放\nEmissions']
    improvement_rates = [15.3, -8.7, 22.1, -12.4]
    colors = ['green' if x > 0 else 'red' for x in improvement_rates]
    
    bars = ax1.bar(mixed_labels, improvement_rates, color=colors, alpha=0.7)
    ax1.set_title('Performance Improvement 性能改善', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement Rate (%) 改善率', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加混合语言数值标签
    for bar, value in zip(bars, improvement_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 测试2: 英文为主的技术指标
    tech_metrics = ['Response Time (ms)', 'Throughput (req/s)', 'CPU Usage (%)', 'Memory (MB)']
    tech_values = [45.2, 1250.8, 67.3, 512.7]
    
    ax2.plot(tech_metrics, tech_values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax2.set_title('Technical Metrics 技术指标', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Measurement Values 测量值', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, (metric, value) in enumerate(zip(tech_metrics, tech_values)):
        ax2.text(i, value + max(tech_values)*0.02, f'{value:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存混合语言测试图片
    mixed_output_path = 'ubuntu_mixed_language_test.png'
    plt.savefig(mixed_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 中英文混合测试图表已保存: {mixed_output_path}")
    
    return mixed_output_path

if __name__ == "__main__":
    try:
        print("🔧 开始Ubuntu英文字体修复测试...")
        
        # 测试英文字体修复
        english_result = test_ubuntu_english_font_fix()
        
        # 测试中英文混合显示
        mixed_result = test_mixed_language_display()
        
        print("\n" + "=" * 60)
        print("Ubuntu英文字体修复测试完成")
        print("=" * 60)
        print("\n✅ 修复内容:")
        print("  • 调整Ubuntu字体优先级")
        print("  • 英文字体 (Liberation Sans, DejaVu Sans) 优先")
        print("  • 中文字体作为备选")
        print("  • 保持负号显示修复")
        print("  • 增强数学文本渲染")
        
        print("\n📊 测试结果:")
        print(f"  • 英文字体测试: {english_result}")
        print(f"  • 混合语言测试: {mixed_result}")
        
        print("\n🎯 预期效果:")
        print("  ✓ Ubuntu系统英文字符正常显示")
        print("  ✓ 中文字符保持正确显示")
        print("  ✓ 数字和符号正确渲染")
        print("  ✓ 混合语言标签清晰可读")
        
        print(f"\n🎉 Ubuntu英文字体修复测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
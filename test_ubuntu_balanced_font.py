#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu中英文平衡字体测试
验证修复后的Ubuntu字体配置能否同时正确显示中英文字符
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

def test_ubuntu_balanced_font():
    """测试Ubuntu中英文平衡字体配置"""
    print("=" * 60)
    print("Ubuntu中英文平衡字体测试")
    print("=" * 60)
    
    # 模拟Ubuntu环境并导入修复后的字体设置
    with patch('platform.system') as mock_system:
        mock_system.return_value = 'Linux'
        
        # 重新导入字体设置函数
        from atscui.utils.comparative_analysis import setup_chinese_fonts
        
        print("\n正在模拟Ubuntu环境并配置平衡字体...")
        selected_font = setup_chinese_fonts()
        
        # 获取当前字体配置
        current_fonts = plt.rcParams['font.sans-serif']
        print(f"\n当前字体配置:")
        print(f"  主要字体: {selected_font}")
        print(f"  字体优先级: {current_fonts[:5]}")
        print(f"  中文字体优先: {'Noto Sans CJK SC' in current_fonts[:3] or 'WenQuanYi' in str(current_fonts[:3])}")
        print(f"  英文字体支持: {'Liberation Sans' in current_fonts or 'DejaVu Sans' in current_fonts}")
        print(f"  负号处理: {not plt.rcParams['axes.unicode_minus']}")
    
    # 创建中英文平衡测试图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ubuntu中英文平衡字体测试 - Ubuntu Balanced Font Test', fontsize=16, fontweight='bold')
    
    # 测试1: 中英文混合标签（重点测试）
    mixed_categories = ['等待时间\nWaiting Time', '通行速度\nTraffic Speed', '燃油消耗\nFuel Usage', 'CO₂排放\nEmissions', '效率提升\nEfficiency']
    values = [25.3, -12.8, 45.7, -8.5, 19.2]
    colors = ['#E74C3C' if v > 0 else '#27AE60' for v in values]
    
    bars = ax1.bar(mixed_categories, values, color=colors, alpha=0.8)
    ax1.set_title('性能指标对比 Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('改善率 (%) Improvement Rate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加中英文数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 测试2: 纯中文标签
    chinese_algorithms = ['强化学习算法', '深度Q网络', '优势行动评判', '软行动评判', '固定配时']
    scores = [85.6, 78.3, 82.1, 79.8, 65.4]
    
    ax2.plot(chinese_algorithms, scores, marker='o', linewidth=2, markersize=8, color='#3498DB')
    ax2.set_title('算法性能对比（中文标签）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('性能得分', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加中文数值标签
    for i, (alg, score) in enumerate(zip(chinese_algorithms, scores)):
        ax2.text(i, score + 1, f'{score:.1f}分', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 测试3: 纯英文标签
    english_metrics = ['Response Time', 'Throughput', 'CPU Usage', 'Memory Usage', 'Network I/O']
    metric_values = [45.2, 123.7, 67.3, 512.7, 89.1]
    
    ax3.barh(english_metrics, metric_values, color='#9B59B6', alpha=0.7)
    ax3.set_title('System Metrics (English Labels)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Measurement Values', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 添加英文数值标签
    for i, value in enumerate(metric_values):
        ax3.text(value + 5, i, f'{value:.1f}', ha='left', va='center',
                fontsize=10, fontweight='bold')
    
    # 测试4: 复杂中英文混合场景
    complex_labels = ['早高峰 Morning\n(7-9时)', '平峰期 Off-Peak\n(9-17时)', '晚高峰 Evening\n(17-19时)', '夜间 Night\n(19-7时)']
    traffic_data = [1200, 800, 1350, 400]
    efficiency_data = [75.5, 85.2, 72.8, 90.1]
    
    # 双Y轴图表
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(complex_labels, traffic_data, marker='s', linewidth=3, markersize=8, 
                     color='#E67E22', label='交通流量 Traffic Flow')
    line2 = ax4_twin.plot(complex_labels, efficiency_data, marker='^', linewidth=3, markersize=8, 
                          color='#27AE60', label='通行效率 Efficiency')
    
    ax4.set_title('日交通模式分析 Daily Traffic Pattern', fontsize=14, fontweight='bold')
    ax4.set_ylabel('流量 (辆/小时) Flow (veh/h)', fontsize=12, color='#E67E22')
    ax4_twin.set_ylabel('效率 (%) Efficiency', fontsize=12, color='#27AE60')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加复合标签
    for i, (time, flow, eff) in enumerate(zip(complex_labels, traffic_data, efficiency_data)):
        ax4.text(i, flow + 30, f'{flow}辆/h', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#E67E22')
        ax4_twin.text(i, eff + 1, f'{eff:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#27AE60')
    
    # 添加图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存测试图片
    output_path = 'ubuntu_balanced_font_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Ubuntu中英文平衡测试图表已保存: {output_path}")
    
    return output_path

def test_special_characters():
    """测试特殊字符和符号显示"""
    print("\n" + "=" * 60)
    print("特殊字符和符号测试")
    print("=" * 60)
    
    # 创建特殊字符测试图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('特殊字符测试 Special Characters Test', fontsize=16, fontweight='bold')
    
    # 测试1: 数学符号和单位
    math_labels = ['速度 (km/h)', '时间 (秒)', '距离 (米)', '角度 (°)', '温度 (℃)', '压力 (Pa)']
    math_values = [45.2, -123.7, 1850, 90, 25, 101325]
    
    bars = ax1.bar(math_labels, math_values, color='#3498DB', alpha=0.7)
    ax1.set_title('数学符号和单位 Math Symbols & Units', fontsize=14, fontweight='bold')
    ax1.set_ylabel('测量值 Measurement Values', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加带符号的数值标签
    for bar, value in zip(bars, math_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (abs(height)*0.02 if height >= 0 else -abs(height)*0.05),
                f'{value:+.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 测试2: 特殊符号和表情
    special_categories = ['优秀 ★★★', '良好 ★★☆', '一般 ★☆☆', '较差 ☆☆☆']
    ratings = [95, 80, 65, 40]
    colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C']
    
    ax2.pie(ratings, labels=special_categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('评级分布 Rating Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存特殊字符测试图片
    special_output_path = 'ubuntu_special_characters_test.png'
    plt.savefig(special_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 特殊字符测试图表已保存: {special_output_path}")
    
    return special_output_path

if __name__ == "__main__":
    try:
        print("🔧 开始Ubuntu中英文平衡字体测试...")
        
        # 测试中英文平衡显示
        balanced_result = test_ubuntu_balanced_font()
        
        # 测试特殊字符显示
        special_result = test_special_characters()
        
        print("\n" + "=" * 60)
        print("Ubuntu中英文平衡字体测试完成")
        print("=" * 60)
        print("\n✅ 修复内容:")
        print("  • 调整Ubuntu字体优先级")
        print("  • 中文字体 (Noto Sans CJK SC) 优先")
        print("  • 英文字体作为备选支持")
        print("  • 保持负号显示修复")
        print("  • 增强数学文本渲染")
        
        print("\n📊 测试结果:")
        print(f"  • 中英文平衡测试: {balanced_result}")
        print(f"  • 特殊字符测试: {special_result}")
        
        print("\n🎯 预期效果:")
        print("  ✓ Ubuntu系统中文字符正常显示")
        print("  ✓ 英文字符保持清晰可读")
        print("  ✓ 中英文混合标签完美渲染")
        print("  ✓ 数字、符号、特殊字符正确显示")
        print("  ✓ 数学公式和单位符号支持")
        
        print(f"\n🎉 Ubuntu中英文平衡字体测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
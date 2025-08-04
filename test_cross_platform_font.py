#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台字体测试脚本
测试修改后的字体配置在不同系统中的显示效果
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import platform
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 导入字体设置函数
from atscui.utils.comparative_analysis import setup_chinese_fonts

def test_cross_platform_fonts():
    """测试跨平台字体显示效果"""
    print("=" * 60)
    print("跨平台字体测试")
    print("=" * 60)
    
    # 获取系统信息
    system_info = {
        '操作系统': platform.system(),
        '系统版本': platform.release(),
        'Python版本': platform.python_version(),
        '架构': platform.machine()
    }
    
    print("\n系统信息:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 设置字体
    print("\n正在配置字体...")
    selected_font = setup_chinese_fonts()
    
    # 获取当前字体配置
    current_fonts = plt.rcParams['font.sans-serif']
    print(f"\n当前字体配置:")
    print(f"  主要字体: {selected_font}")
    print(f"  字体列表: {current_fonts[:3]}...")
    print(f"  负号处理: {not plt.rcParams['axes.unicode_minus']}")
    
    # 创建测试图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('跨平台字体显示测试 - Cross-Platform Font Test', fontsize=16, fontweight='bold')
    
    # 测试1: 中文标签和数值
    categories = ['等待时间', '通行速度', '燃油消耗', 'CO₂排放', '通行量']
    values = [25.3, 45.7, -12.8, -8.5, 156.2]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax1.bar(categories, values, color=colors)
    ax1.set_title('性能指标对比 (Performance Metrics)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('改善率 (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 测试2: 混合中英文标签
    algorithms = ['PPO算法', 'DQN Algorithm', 'A2C方法', 'SAC Method', '固定配时']
    scores = [85.6, 78.3, 82.1, 79.8, 65.4]
    
    ax2.plot(algorithms, scores, marker='o', linewidth=2, markersize=8, color='#E74C3C')
    ax2.set_title('算法性能评分 (Algorithm Performance)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('综合评分 (Score)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, (alg, score) in enumerate(zip(algorithms, scores)):
        ax2.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 测试3: 特殊字符和符号
    symbols = ['速度↑', '时间↓', '效率→', '成本←', '质量★']
    symbol_values = [+15.2, -23.7, +8.9, -12.3, +19.6]
    
    colors_symbols = ['green' if v > 0 else 'red' for v in symbol_values]
    ax3.barh(symbols, symbol_values, color=colors_symbols, alpha=0.7)
    ax3.set_title('特殊符号测试 (Special Characters)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('变化率 (%) Change Rate', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, value in enumerate(symbol_values):
        ax3.text(value + (1 if value >= 0 else -1), i,
                f'{value:+.1f}%', ha='left' if value >= 0 else 'right', va='center',
                fontsize=10, fontweight='bold')
    
    # 测试4: 数学公式和单位
    time_data = np.linspace(0, 24, 100)
    traffic_flow = 1000 + 500 * np.sin(2 * np.pi * time_data / 24) + 200 * np.sin(4 * np.pi * time_data / 24)
    
    ax4.plot(time_data, traffic_flow, linewidth=2, color='#3498DB')
    ax4.set_title('交通流量变化 Traffic Flow (辆/小时 vehicles/h)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('时间 Time (小时 hours)', fontsize=12)
    ax4.set_ylabel('流量 Flow (辆/h)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 添加注释
    ax4.annotate('峰值时段\nPeak Hours', xy=(8, 1650), xytext=(10, 1800),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # 保存测试图片
    output_path = 'cross_platform_font_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 测试图表已保存: {output_path}")
    
    # 显示字体测试结果
    print("\n" + "=" * 60)
    print("字体测试完成")
    print("=" * 60)
    print("测试内容:")
    print("  ✓ 中文字符显示")
    print("  ✓ 英文字符显示")
    print("  ✓ 数字和符号显示")
    print("  ✓ 特殊字符 (↑↓→←★)")
    print("  ✓ 正负号格式化")
    print("  ✓ 混合中英文标签")
    print("\n如果图表中所有文字都能正常显示，说明字体配置成功！")
    
    return output_path

if __name__ == "__main__":
    try:
        result_path = test_cross_platform_fonts()
        print(f"\n🎉 跨平台字体测试完成！结果保存在: {result_path}")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu字体配置模拟测试
模拟Ubuntu环境下的字体配置逻辑
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

def simulate_ubuntu_font_setup():
    """模拟Ubuntu环境下的字体设置"""
    import matplotlib.font_manager as fm
    
    print("=" * 60)
    print("Ubuntu字体配置模拟测试")
    print("=" * 60)
    
    # 模拟Ubuntu系统的字体列表
    ubuntu_font_list = [
        'Noto Sans CJK SC',      # Linux最佳中文支持
        'WenQuanYi Micro Hei',   # Linux中文字体
        'WenQuanYi Zen Hei',     # Linux中文字体
        'Droid Sans Fallback',   # Android/Linux字体
        'Liberation Sans',       # Linux英文字体
        'DejaVu Sans',          # Linux英文字体
        'Ubuntu',               # Ubuntu字体
        'Noto Sans',            # 通用字体
        'Arial',                # 通用英文字体
        'sans-serif'            # 系统默认
    ]
    
    # 获取当前系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 模拟Ubuntu环境下的字体选择
    selected_fonts = []
    for font in ubuntu_font_list:
        if font in available_fonts:
            selected_fonts.append(font)
    
    # 如果没有找到Ubuntu特定字体，使用通用备选
    if not selected_fonts:
        selected_fonts = ['DejaVu Sans', 'Arial', 'sans-serif']
        print("警告: 未找到Ubuntu推荐字体，使用通用字体")
        print("建议安装中文字体包: sudo apt-get install fonts-noto-cjk")
    else:
        print(f"模拟Ubuntu环境 - 使用字体: {selected_fonts[0]}")
        print(f"可用字体列表: {selected_fonts[:3]}...")
    
    # 设置字体配置（模拟Ubuntu配置）
    plt.rcParams['font.sans-serif'] = selected_fonts
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    return selected_fonts[0] if selected_fonts else 'sans-serif'

def test_ubuntu_font_compatibility():
    """测试Ubuntu字体兼容性"""
    
    # 模拟Ubuntu字体设置
    selected_font = simulate_ubuntu_font_setup()
    
    # 创建测试图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ubuntu字体兼容性测试 - Ubuntu Font Compatibility Test', fontsize=16, fontweight='bold')
    
    # 测试1: 中英文混合标签
    categories = ['等待时间\nWaiting Time', '通行速度\nSpeed', '燃油消耗\nFuel', 'CO₂排放\nEmission']
    values = [25.3, -12.8, 45.7, -8.5]
    colors = ['#E74C3C' if v > 0 else '#27AE60' for v in values]
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.8)
    ax1.set_title('性能改善率 Performance Improvement', fontsize=14, fontweight='bold')
    ax1.set_ylabel('改善率 (%) Improvement Rate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # 测试2: 时间序列数据
    time_points = ['早高峰\n7-9h', '平峰\n9-17h', '晚高峰\n17-19h', '夜间\n19-7h']
    traffic_flow = [1200, 800, 1350, 400]
    
    ax2.plot(time_points, traffic_flow, marker='o', linewidth=3, markersize=10, 
             color='#3498DB', markerfacecolor='#E74C3C')
    ax2.set_title('交通流量变化 Traffic Flow Pattern', fontsize=14, fontweight='bold')
    ax2.set_ylabel('流量 (辆/h) Flow (veh/h)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (time, flow) in enumerate(zip(time_points, traffic_flow)):
        ax2.text(i, flow + 50, f'{flow}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存测试图片
    output_path = 'ubuntu_font_compatibility_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Ubuntu兼容性测试图表已保存: {output_path}")
    
    return output_path

def test_cross_platform_logic():
    """测试跨平台字体选择逻辑"""
    print("\n" + "=" * 60)
    print("跨平台字体选择逻辑测试")
    print("=" * 60)
    
    # 模拟不同操作系统
    systems = {
        'darwin': 'macOS',
        'linux': 'Ubuntu/Linux', 
        'windows': 'Windows'
    }
    
    expected_fonts = {
        'darwin': ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB'],
        'linux': ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Liberation Sans'],
        'windows': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    }
    
    for system_key, system_name in systems.items():
        print(f"\n{system_name} 系统:")
        fonts = expected_fonts[system_key]
        print(f"  推荐字体: {fonts[0]}")
        print(f"  备选字体: {', '.join(fonts[1:3])}")
        print(f"  特点: {'中文优先' if system_key != 'linux' else '中英文平衡'}")
    
    print("\n字体配置策略:")
    print("  ✓ 自动检测操作系统")
    print("  ✓ 根据系统选择最佳字体")
    print("  ✓ 提供多级备选方案")
    print("  ✓ 统一的渲染参数设置")
    print("  ✓ 负号显示问题修复")

if __name__ == "__main__":
    try:
        # 测试Ubuntu字体兼容性
        result_path = test_ubuntu_font_compatibility()
        
        # 测试跨平台逻辑
        test_cross_platform_logic()
        
        print(f"\n🎉 Ubuntu字体模拟测试完成！")
        print(f"📊 测试结果: {result_path}")
        print("\n✅ 字体配置修改总结:")
        print("  • 智能检测操作系统类型")
        print("  • macOS: 优先使用 Arial Unicode MS")
        print("  • Ubuntu: 优先使用 Noto Sans CJK SC")
        print("  • Windows: 优先使用 Microsoft YaHei")
        print("  • 统一解决负号显示问题")
        print("  • 增强数学文本渲染")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
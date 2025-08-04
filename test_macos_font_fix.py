#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试macOS字体修复效果
用于验证macOS系统中的中文字符显示是否正常
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from atscui.utils.comparative_analysis import setup_chinese_fonts

def test_macos_font_display():
    """
    测试macOS字体显示效果
    """
    print("开始测试macOS字体显示效果...")
    
    # 初始化字体设置
    selected_font = setup_chinese_fonts()
    
    # 创建测试图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 测试1: 中文标签和数值
    algorithms = ['PPO算法', 'DQN算法', 'A2C算法', 'SAC算法']
    scores = [92.5, 88.3, 85.7, 90.1]
    
    bars = ax1.bar(algorithms, scores, color=['#2E8B57', '#4169E1', '#FF6347', '#32CD32'], alpha=0.8)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}分', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('算法性能对比测试', fontsize=14, fontweight='bold')
    ax1.set_ylabel('性能评分')
    ax1.set_xlabel('强化学习算法')
    ax1.set_ylim(0, 100)
    
    # 测试2: 交通指标中文显示
    metrics = ['平均等待时间', '车辆通行量', '路口效率', '燃油消耗']
    values = [45.2, 1250, 78.9, 32.1]
    colors = ['#FF69B4', '#20B2AA', '#FFD700', '#FF4500']
    
    bars2 = ax2.bar(metrics, values, color=colors, alpha=0.8)
    
    for i, (bar, value) in enumerate(zip(bars2, values)):
        height = bar.get_height()
        if '时间' in metrics[i]:
            unit = '秒'
        elif '通行量' in metrics[i]:
            unit = '辆/小时'
        elif '效率' in metrics[i]:
            unit = '%'
        else:
            unit = 'L/km'
        
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{value:.1f}{unit}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('交通指标评估结果', fontsize=14, fontweight='bold')
    ax2.set_ylabel('指标数值')
    ax2.set_xlabel('评估指标')
    
    # 旋转x轴标签以避免重叠
    ax1.tick_params(axis='x', rotation=15)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # 保存测试图表
    output_path = '/Users/xnpeng/sumoptis/atscui/outs/macos_font_test.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ macOS字体测试图表已保存: {output_path}")
    print("\n字体配置信息:")
    print(f"选择的字体: {selected_font}")
    print("\n修复说明:")
    print("1. 恢复了macOS优化的字体优先级列表")
    print("2. 中文字体优先，Arial Unicode MS作为首选")
    print("3. 移除了Ubuntu特定的字体设置")
    print("4. 简化了字体配置，避免冲突")
    print("\n请检查生成的测试图表，确认中文字符显示正常")
    
    return output_path

def test_font_availability():
    """
    检查macOS系统可用字体
    """
    import matplotlib.font_manager as fm
    
    print("\n=== macOS系统字体检查 ===")
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 检查关键字体
    key_fonts = [
        'Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti',
        'Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei'
    ]
    
    for font in key_fonts:
        status = "✅ 可用" if font in available_fonts else "❌ 不可用"
        print(f"{font}: {status}")
    
    print(f"\n系统总共有 {len(available_fonts)} 个字体可用")

if __name__ == "__main__":
    try:
        # 测试字体可用性
        test_font_availability()
        
        # 测试macOS字体显示
        result_path = test_macos_font_display()
        
        print(f"\n🎉 macOS字体修复测试完成！")
        print(f"测试结果保存在: {result_path}")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
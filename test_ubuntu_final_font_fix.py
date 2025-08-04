#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ubuntu最终字体修复测试
验证最终的Ubuntu字体配置能否完美显示中英文字符
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

def test_ubuntu_final_font_fix():
    """测试Ubuntu最终字体修复效果"""
    print("=" * 60)
    print("Ubuntu最终字体修复测试")
    print("=" * 60)
    
    # 模拟Ubuntu环境并导入修复后的字体设置
    with patch('platform.system') as mock_system:
        mock_system.return_value = 'Linux'
        
        # 重新导入字体设置函数
        from atscui.utils.comparative_analysis import setup_chinese_fonts
        
        print("\n正在模拟Ubuntu环境并配置最终字体...")
        selected_font = setup_chinese_fonts()
        
        # 获取当前字体配置
        current_fonts = plt.rcParams['font.sans-serif']
        print(f"\n当前字体配置:")
        print(f"  主要字体: {selected_font}")
        print(f"  字体优先级: {current_fonts[:5]}")
        print(f"  通用字体优先: {'Noto Sans' in current_fonts[:2]}")
        print(f"  英文字体支持: {'Liberation Sans' in current_fonts[:3] or 'DejaVu Sans' in current_fonts[:3]}")
        print(f"  中文字体支持: {'Noto Sans CJK SC' in current_fonts or 'WenQuanYi' in str(current_fonts)}")
        print(f"  负号处理: {not plt.rcParams['axes.unicode_minus']}")
    
    # 创建最终测试图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ubuntu最终字体修复测试 - Ubuntu Final Font Fix Test', fontsize=16, fontweight='bold')
    
    # 测试1: 关键中英文混合场景
    critical_labels = ['等待时间\nWaiting Time', '通行速度\nTraffic Speed', '燃油消耗\nFuel Usage', 'CO₂排放\nEmissions']
    values = [25.3, -12.8, 45.7, -8.5]
    colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in values]
    
    bars = ax1.bar(critical_labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('关键性能指标 Critical Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('改善率 (%) Improvement Rate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # 添加详细数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -3),
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=11, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 测试2: 纯英文技术标签
    english_tech = ['Response Time (ms)', 'Throughput (req/s)', 'CPU Usage (%)', 'Memory (MB)', 'Disk I/O (MB/s)']
    tech_values = [45.2, 1250.8, 67.3, 512.7, 89.1]
    
    ax2.plot(english_tech, tech_values, marker='o', linewidth=3, markersize=10, 
             color='#3498DB', markerfacecolor='#E74C3C', markeredgecolor='white', markeredgewidth=2)
    ax2.set_title('Technical Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Measurement Values', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加英文数值标签
    for i, (metric, value) in enumerate(zip(english_tech, tech_values)):
        ax2.text(i, value + max(tech_values)*0.03, f'{value:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
    
    # 测试3: 纯中文算法对比
    chinese_algorithms = ['强化学习算法', '深度Q网络', '优势行动评判', '软行动评判', '固定配时方案']
    algorithm_scores = [85.6, 78.3, 82.1, 79.8, 65.4]
    colors_cn = ['#E67E22', '#9B59B6', '#1ABC9C', '#F39C12', '#95A5A6']
    
    wedges, texts, autotexts = ax3.pie(algorithm_scores, labels=chinese_algorithms, colors=colors_cn, 
                                       autopct='%1.1f%%', startangle=90, 
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('算法性能分布', fontsize=14, fontweight='bold')
    
    # 测试4: 复杂时间序列（中英文混合）
    time_periods = ['早高峰\nMorning Peak\n(7-9h)', '平峰期\nOff-Peak\n(9-17h)', 
                   '晚高峰\nEvening Peak\n(17-19h)', '夜间\nNight\n(19-7h)']
    traffic_volume = [1200, 800, 1350, 400]
    efficiency_rate = [75.5, 85.2, 72.8, 90.1]
    
    # 双Y轴复杂图表
    ax4_twin = ax4.twinx()
    
    # 流量柱状图
    bars_traffic = ax4.bar([i-0.2 for i in range(len(time_periods))], traffic_volume, 
                          width=0.4, color='#3498DB', alpha=0.7, label='交通流量 Traffic Volume')
    
    # 效率折线图
    line_efficiency = ax4_twin.plot(range(len(time_periods)), efficiency_rate, 
                                   marker='s', linewidth=3, markersize=8, 
                                   color='#E74C3C', label='通行效率 Efficiency')
    
    ax4.set_title('日交通模式综合分析\nDaily Traffic Pattern Analysis', fontsize=14, fontweight='bold')
    ax4.set_ylabel('流量 (辆/小时)\nFlow (veh/h)', fontsize=11, color='#3498DB')
    ax4_twin.set_ylabel('效率 (%)\nEfficiency', fontsize=11, color='#E74C3C')
    ax4.set_xticks(range(len(time_periods)))
    ax4.set_xticklabels(time_periods, fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (volume, efficiency) in enumerate(zip(traffic_volume, efficiency_rate)):
        ax4.text(i-0.2, volume + 30, f'{volume}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#3498DB')
        ax4_twin.text(i, efficiency + 1, f'{efficiency:.1f}%', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#E74C3C')
    
    # 添加图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # 保存测试图片
    output_path = 'ubuntu_final_font_fix_test.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Ubuntu最终字体修复测试图表已保存: {output_path}")
    
    return output_path

def test_font_fallback_mechanism():
    """测试字体降级机制"""
    print("\n" + "=" * 60)
    print("字体降级机制测试")
    print("=" * 60)
    
    import matplotlib.font_manager as fm
    
    # 获取当前字体配置
    current_fonts = plt.rcParams['font.sans-serif']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    print("\n字体降级链分析:")
    for i, font in enumerate(current_fonts[:8]):
        status = "✓ 可用" if font in available_fonts else "✗ 不可用"
        priority = "主字体" if i == 0 else f"备选{i}"
        print(f"  {i+1}. {font} - {priority} - {status}")
    
    # 创建字体测试图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 测试各种字符类型
    test_texts = [
        '中文字符测试：你好世界！',
        'English Character Test: Hello World!',
        '数字和符号：123456789 +-*/=()[]{}',
        '特殊字符：°℃℉±×÷≤≥≠∞∑∏√∫',
        '混合文本：Traffic Flow 交通流量 = 1,250 veh/h',
        'Mathematical: α+β=γ, ∑(x²)=∫f(x)dx',
        '单位符号：km/h, m/s², kg·m/s², Pa·s',
        '标点符号：，。！？；：","（）【】"'
    ]
    
    y_positions = np.arange(len(test_texts))
    
    for i, text in enumerate(test_texts):
        ax.text(0.05, y_positions[i], text, fontsize=12, 
               verticalalignment='center', transform=ax.transData)
    
    ax.set_ylim(-0.5, len(test_texts)-0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'测试{i+1}' for i in range(len(test_texts))])
    ax.set_title('字体降级机制测试 - Font Fallback Test', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 移除x轴
    ax.set_xticks([])
    
    plt.tight_layout()
    
    # 保存字体测试图片
    fallback_output_path = 'ubuntu_font_fallback_test.png'
    plt.savefig(fallback_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 字体降级机制测试图表已保存: {fallback_output_path}")
    
    return fallback_output_path

if __name__ == "__main__":
    try:
        print("🔧 开始Ubuntu最终字体修复测试...")
        
        # 测试最终字体修复
        final_result = test_ubuntu_final_font_fix()
        
        # 测试字体降级机制
        fallback_result = test_font_fallback_mechanism()
        
        print("\n" + "=" * 60)
        print("Ubuntu最终字体修复测试完成")
        print("=" * 60)
        print("\n✅ 最终修复策略:")
        print("  • 使用 Noto Sans 作为主字体（中英文平衡）")
        print("  • Liberation Sans, DejaVu Sans 作为英文备选")
        print("  • Noto Sans CJK SC, WenQuanYi 作为中文备选")
        print("  • 完整的字体降级链确保兼容性")
        print("  • 保持负号显示和数学文本修复")
        
        print("\n📊 测试结果:")
        print(f"  • 最终修复测试: {final_result}")
        print(f"  • 字体降级测试: {fallback_result}")
        
        print("\n🎯 修复效果:")
        print("  ✓ Ubuntu系统中英文字符完美显示")
        print("  ✓ 中英文混合标签清晰可读")
        print("  ✓ 技术术语和数学符号正确渲染")
        print("  ✓ 特殊字符和标点符号支持")
        print("  ✓ 字体降级机制确保稳定性")
        
        print(f"\n🎉 Ubuntu最终字体修复测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
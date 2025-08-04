#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体警告修复验证测试
验证修复后的字体配置能否消除中文字符缺失警告
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from unittest.mock import patch
import warnings

# 添加项目路径
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_font_warning_fix():
    """测试字体警告修复效果"""
    print("=" * 60)
    print("字体警告修复验证测试")
    print("=" * 60)
    
    # 捕获警告信息
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # 模拟Ubuntu环境并导入修复后的字体设置
        with patch('platform.system') as mock_system:
            mock_system.return_value = 'Linux'
            
            # 重新导入字体设置函数
            from atscui.utils.comparative_analysis import setup_chinese_fonts
            
            print("\n正在模拟Ubuntu环境并配置修复后的字体...")
            selected_font = setup_chinese_fonts()
            
            # 获取当前字体配置
            current_fonts = plt.rcParams['font.sans-serif']
            print(f"\n当前字体配置:")
            print(f"  主要字体: {selected_font}")
            print(f"  字体优先级: {current_fonts[:5]}")
            print(f"  中文字体优先: {'Noto Sans CJK SC' in current_fonts[:2] or 'WenQuanYi' in str(current_fonts[:2])}")
            print(f"  负号处理: {not plt.rcParams['axes.unicode_minus']}")
    
        # 创建包含中文字符的测试图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('字体警告修复验证 - Font Warning Fix Verification', fontsize=16, fontweight='bold')
        
        # 测试1: 包含问题字符的标签
        problematic_labels = ['配置 Config', '重量 Weight', '速度 Speed', '效率 Efficiency']
        values = [85.3, 92.1, 78.6, 88.9]
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        
        bars = ax1.bar(problematic_labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('包含中文字符的测试 Chinese Character Test', fontsize=14, fontweight='bold')
        ax1.set_ylabel('数值 Values (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        
        # 测试2: 复杂中文字符
        complex_chinese = ['交通流量\nTraffic Flow', '信号配时\nSignal Timing', 
                          '路口优化\nIntersection Opt', '算法性能\nAlgorithm Perf']
        performance_data = [1250, 980, 1100, 1350]
        
        ax2.plot(range(len(complex_chinese)), performance_data, 
                marker='o', linewidth=3, markersize=10, color='#9B59B6',
                markerfacecolor='#E67E22', markeredgecolor='white', markeredgewidth=2)
        ax2.set_title('复杂中文字符测试 Complex Chinese Characters', fontsize=14, fontweight='bold')
        ax2.set_ylabel('性能指标 Performance', fontsize=12)
        ax2.set_xticks(range(len(complex_chinese)))
        ax2.set_xticklabels(complex_chinese, fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 测试3: 特殊中文字符和符号
        special_chars = ['温度°C', '压力±Pa', '速度≥50km/h', '效率≤90%', '距离∞m']
        special_values = [25.5, 101.3, 65.2, 87.8, 999.9]
        
        wedges, texts, autotexts = ax3.pie(special_values, labels=special_chars, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax3.set_title('特殊字符和符号测试 Special Characters Test', fontsize=14, fontweight='bold')
        
        # 测试4: 混合语言时间序列
        time_labels = ['早晨\nMorning\n6-9时', '上午\nAM\n9-12时', 
                      '下午\nPM\n12-18时', '晚上\nEvening\n18-24时']
        traffic_data = [800, 600, 1200, 400]
        efficiency_data = [75, 85, 70, 90]
        
        # 双Y轴图表
        ax4_twin = ax4.twinx()
        
        # 流量柱状图
        bars_traffic = ax4.bar([i-0.2 for i in range(len(time_labels))], traffic_data,
                              width=0.4, color='#3498DB', alpha=0.7, label='流量 Flow')
        
        # 效率折线图
        line_efficiency = ax4_twin.plot(range(len(time_labels)), efficiency_data,
                                       marker='s', linewidth=3, markersize=8,
                                       color='#E74C3C', label='效率 Efficiency')
        
        ax4.set_title('混合语言时间序列 Mixed Language Time Series', fontsize=14, fontweight='bold')
        ax4.set_ylabel('流量 Flow (veh/h)', fontsize=11, color='#3498DB')
        ax4_twin.set_ylabel('效率 Efficiency (%)', fontsize=11, color='#E74C3C')
        ax4.set_xticks(range(len(time_labels)))
        ax4.set_xticklabels(time_labels, fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # 保存测试图片
        output_path = 'font_warning_fix_verification.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✅ 字体警告修复验证图表已保存: {output_path}")
        
        # 检查警告信息
        font_warnings = [warning for warning in w if 'missing from font' in str(warning.message)]
        
        print(f"\n📊 警告检查结果:")
        if font_warnings:
            print(f"  ❌ 仍有 {len(font_warnings)} 个字体警告:")
            for warning in font_warnings[:3]:  # 只显示前3个
                print(f"    • {warning.message}")
            if len(font_warnings) > 3:
                print(f"    • ... 还有 {len(font_warnings)-3} 个警告")
        else:
            print(f"  ✅ 无字体缺失警告！字体配置修复成功")
        
        return output_path, len(font_warnings)

def test_font_availability():
    """测试字体可用性"""
    print("\n" + "=" * 60)
    print("字体可用性检查")
    print("=" * 60)
    
    import matplotlib.font_manager as fm
    
    # 获取所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 检查关键中文字体
    key_chinese_fonts = [
        'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 
        'WenQuanYi Zen Hei',
        'Noto Sans',
        'Liberation Sans',
        'DejaVu Sans'
    ]
    
    print("\n关键字体可用性检查:")
    available_count = 0
    for font in key_chinese_fonts:
        status = "✓ 可用" if font in available_fonts else "✗ 不可用"
        print(f"  {font}: {status}")
        if font in available_fonts:
            available_count += 1
    
    print(f"\n📈 字体可用性统计:")
    print(f"  • 关键字体可用: {available_count}/{len(key_chinese_fonts)}")
    print(f"  • 系统总字体数: {len(available_fonts)}")
    
    # 检查中文字体支持
    chinese_fonts = [f for f in available_fonts if any(keyword in f for keyword in 
                    ['CJK', 'Chinese', 'WenQuanYi', 'Noto', 'SimHei', 'Microsoft YaHei'])]
    print(f"  • 中文相关字体: {len(chinese_fonts)}")
    
    if chinese_fonts:
        print(f"\n🔤 检测到的中文字体:")
        for font in chinese_fonts[:5]:  # 显示前5个
            print(f"    • {font}")
        if len(chinese_fonts) > 5:
            print(f"    • ... 还有 {len(chinese_fonts)-5} 个中文字体")
    
    return available_count, len(chinese_fonts)

if __name__ == "__main__":
    try:
        print("🔧 开始字体警告修复验证测试...")
        
        # 测试字体可用性
        available_count, chinese_count = test_font_availability()
        
        # 测试字体警告修复
        result_path, warning_count = test_font_warning_fix()
        
        print("\n" + "=" * 60)
        print("字体警告修复验证完成")
        print("=" * 60)
        
        print("\n✅ 修复策略总结:")
        print("  • 将中文字体设置为最高优先级")
        print("  • Noto Sans CJK SC 作为主要中文字体")
        print("  • WenQuanYi 系列作为中文备选")
        print("  • 保持英文字体作为降级选项")
        
        print(f"\n📊 测试结果:")
        print(f"  • 验证图表: {result_path}")
        print(f"  • 关键字体可用: {available_count}/6")
        print(f"  • 中文字体数量: {chinese_count}")
        print(f"  • 字体警告数量: {warning_count}")
        
        print(f"\n🎯 修复效果:")
        if warning_count == 0:
            print("  ✅ 字体警告完全消除")
            print("  ✅ 中文字符正常显示")
            print("  ✅ 字体配置优化成功")
        else:
            print(f"  ⚠️  仍有 {warning_count} 个字体警告")
            print("  💡 建议安装更多中文字体包")
        
        print(f"\n🎉 字体警告修复验证测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
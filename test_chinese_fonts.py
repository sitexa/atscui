#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体测试脚本
用于验证matplotlib中文字体配置是否正常
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def test_chinese_fonts():
    """测试中文字体显示"""
    
    # 设置中文字体
    font_list = [
        'Noto Sans CJK SC',  # Ubuntu推荐的中文字体
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'WenQuanYi Zen Hei',  # 文泉驿正黑
        'Droid Sans Fallback',  # Android字体
        'SimHei',  # Windows字体
        'Arial Unicode MS',  # macOS字体
        'DejaVu Sans'  # 默认字体
    ]
    
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    print("=== 中文字体检测结果 ===")
    selected_font = None
    for font in font_list:
        if font in available_fonts:
            print(f"✓ {font} - 可用")
            if selected_font is None:
                selected_font = font
        else:
            print(f"✗ {font} - 不可用")
    
    if selected_font:
        print(f"\n当前使用字体: {selected_font}")
    else:
        print("\n警告: 未找到合适的中文字体!")
        print("Ubuntu系统建议安装:")
        print("sudo apt-get install fonts-noto-cjk")
        print("sudo apt-get install fonts-wqy-microhei")
        print("sudo apt-get install fonts-wqy-zenhei")
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    x = np.arange(5)
    y = [20, 35, 30, 35, 27]
    
    # 绘制柱状图
    bars = ax.bar(x, y, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # 设置中文标签
    ax.set_xlabel('时间段', fontsize=12)
    ax.set_ylabel('平均等待时间 (秒)', fontsize=12)
    ax.set_title('交通信号控制效果对比分析', fontsize=14, fontweight='bold')
    
    # 设置x轴标签
    labels = ['早高峰', '上午', '中午', '下午', '晚高峰']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # 添加数值标签
    for bar, value in zip(bars, y):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}秒', ha='center', va='bottom')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(y) + 5)
    
    plt.tight_layout()
    
    # 保存图片
    output_file = '/tmp/chinese_font_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n测试图表已保存到: {output_file}")
    print("请检查图表中的中文是否正常显示")
    
    # 显示图表（如果在图形界面环境中）
    try:
        plt.show()
    except:
        print("注意: 当前环境无法显示图形界面")
    
    return selected_font is not None

if __name__ == "__main__":
    print("开始测试中文字体配置...")
    success = test_chinese_fonts()
    
    if success:
        print("\n✓ 中文字体配置成功!")
    else:
        print("\n✗ 中文字体配置失败，请安装相应字体包")
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
from matplotlib import font_manager, rc

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
rc('font', family='Noto Sans CJK SC')  # 添加后就能识别了

# 设置中文字体
matplotlib.rcParams['font.family'] = 'Noto Sans CJK SC'  # 简体中文支持
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 测试绘图
plt.title("测试：横向对比")  # 横 字就是 U+6A2A
plt.xlabel("横轴")
plt.ylabel("纵轴")
plt.plot([1, 2, 3], [3, 2, 5])
plt.show()
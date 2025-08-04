import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

# 1. 获取字体文件路径（相对路径或绝对路径均可）
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

font_path = os.path.join(base_dir, "fonts", "NotoSansCJKsc-Regular.otf")
# 手动注册字体文件
font_manager.fontManager.addfont(font_path)

# 获取字体名称
font_name = font_manager.FontProperties(fname=font_path).get_name()
print("字体名：", font_name)  # 通常是 "Noto Sans CJK SC"


# 3. 设置 matplotlib 全局字体（无衬线字体）为加载的字体名
mpl.rcParams['font.sans-serif'] = [font_name]
# 4. 解决坐标轴负号显示问题
mpl.rcParams['axes.unicode_minus'] = False

# 5. 正常绘图，无需每次写 fontproperties 参数
plt.plot([1, 2, 3], [3, 2, 5])
plt.title("macOS 中文测试")
plt.xlabel("横轴")
plt.ylabel("纵轴")
plt.show()

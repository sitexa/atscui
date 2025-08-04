import matplotlib.pyplot as plt
from matplotlib import font_manager

# 强制使用 .ttc 字体文件路径
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=font_path)

plt.plot([1, 2, 3], [3, 2, 5])
plt.title("中文标题：纵向测试", fontproperties=font_prop)
plt.xlabel("横轴", fontproperties=font_prop)
plt.ylabel("纵轴", fontproperties=font_prop)
plt.show()

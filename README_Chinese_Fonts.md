# Ubuntu系统中文字体配置指南

## 问题描述
在Ubuntu系统中使用matplotlib绘制包含中文的图表时，可能会出现中文乱码问题。这是因为系统缺少合适的中文字体。

## 解决方案

### 1. 安装中文字体包

#### 方法一：安装Noto字体（推荐）
```bash
sudo apt-get update
sudo apt-get install fonts-noto-cjk
```

#### 方法二：安装文泉驿字体
```bash
sudo apt-get install fonts-wqy-microhei
sudo apt-get install fonts-wqy-zenhei
```

#### 方法三：安装所有常用中文字体
```bash
sudo apt-get install fonts-noto-cjk fonts-wqy-microhei fonts-wqy-zenhei
```

### 2. 清除matplotlib字体缓存
安装字体后，需要清除matplotlib的字体缓存：

```bash
# 删除matplotlib字体缓存
rm -rf ~/.cache/matplotlib

# 或者使用Python命令
python3 -c "import matplotlib.font_manager; matplotlib.font_manager._rebuild()"
```

### 3. 验证字体安装

运行测试脚本验证字体是否正确安装：

```bash
cd /Users/xnpeng/sumoptis/atscui
python3 test_chinese_fonts.py
```

### 4. 代码修改说明

在 `atscui/utils/comparative_analysis.py` 中，我们已经修改了字体配置：

```python
# 原始配置（可能在Ubuntu中不工作）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']

# 新配置（兼容Ubuntu系统）
font_list = [
    'Noto Sans CJK SC',      # Ubuntu推荐的中文字体
    'WenQuanYi Micro Hei',   # 文泉驿微米黑
    'WenQuanYi Zen Hei',     # 文泉驿正黑
    'Droid Sans Fallback',   # Android字体
    'SimHei',                # Windows字体
    'Arial Unicode MS',      # macOS字体
    'DejaVu Sans'            # 默认字体
]
plt.rcParams['font.sans-serif'] = font_list
```

## 字体优先级说明

1. **Noto Sans CJK SC** - Google开发的开源中文字体，Ubuntu官方推荐
2. **WenQuanYi Micro Hei** - 文泉驿微米黑，轻量级中文字体
3. **WenQuanYi Zen Hei** - 文泉驿正黑，标准中文字体
4. **Droid Sans Fallback** - Android系统字体
5. **SimHei** - Windows系统字体
6. **Arial Unicode MS** - macOS系统字体
7. **DejaVu Sans** - 默认备用字体

## 常见问题

### Q: 安装字体后仍然显示乱码？
A: 请清除matplotlib字体缓存并重启Python程序：
```bash
rm -rf ~/.cache/matplotlib
```

### Q: 如何检查系统中已安装的中文字体？
A: 使用以下命令：
```bash
fc-list :lang=zh
```

### Q: 在Docker容器中如何解决？
A: 在Dockerfile中添加：
```dockerfile
RUN apt-get update && apt-get install -y fonts-noto-cjk
```

### Q: 如何在代码中动态检测可用字体？
A: 参考 `setup_chinese_fonts()` 函数的实现，它会自动检测并选择可用的中文字体。

## 测试验证

运行测试脚本后，检查以下内容：

1. 控制台输出显示找到可用的中文字体
2. 生成的测试图片 `/tmp/chinese_font_test.png` 中中文显示正常
3. 图表标题、坐标轴标签、数据标签都能正确显示中文

如果测试通过，说明中文字体配置成功，`comparative_analysis.py` 中的图表将能正确显示中文内容。
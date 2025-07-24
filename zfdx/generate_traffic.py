import xml.etree.ElementTree as ET
from xml.dom import minidom # 用于美化输出的XML格式
import os
import random # 引入random模块用于生成随机数
from pathlib import Path

def generate_rou_file(output_filename, flow_type='perhour', total_duration=3600,
                      time_profiles=None, random_factor=0.0, template_file=None):
    """
    动态生成一个 .rou.xml 文件，用于简单的交叉路口仿真。
    引入时间变化和随机性。

    Args:
        output_filename (str): 输出的 .rou.xml 文件名。
        flow_type (str): 流量类型，'perhour' 表示每小时车辆数，'probability' 表示每秒生成概率。
        total_duration (int): 总仿真时长（秒）。
        time_profiles (list): 时间剖面列表，每个元素为 (start_time, end_time, multiplier)。
                              multiplier 用于调整该时间段内的流量。
                              如果为 None，则流量在整个 duration 内保持恒定。
                              例如：[(0, 3600, 1.0), (3600, 7200, 1.5)]
        random_factor (float): 随机波动因子，0.0 表示无随机性，0.1 表示 ±10% 的随机波动。
        template_file (str): 可选的模板文件路径，用于提取路线定义。如果提供，将从该文件中提取路线信息。
    """
    routes_element = ET.Element('routes')

    # 1. 定义车辆类型 (vType)
    ET.SubElement(routes_element, 'vType', {
        'id': 'standard_car',
        'accel': '1.0',
        'decel': '4.5',
        'length': '5.0',
        'minGap': '2.5',
        'maxSpeed': '30',
        'sigma': '0.5'
    })

    # 2. 定义路线 (route)
    # 优先从模板文件提取，否则使用默认示例路线
    if template_file and Path(template_file).exists():
        try:
            # 从模板文件提取路线
            template_tree = ET.parse(template_file)
            template_root = template_tree.getroot()
            
            example_routes = {}
            for route in template_root.findall('route'):
                route_id = route.get('id')
                edges = route.get('edges')
                if route_id and edges:
                    example_routes[route_id] = edges
                    
            print(f"从模板文件 {template_file} 提取到 {len(example_routes)} 条路线")
            
        except Exception as e:
            print(f"从模板文件提取路线失败: {e}，使用默认路线")
            # 使用默认示例路线
            example_routes = {
                'route_ns': 'n_t t_s', # 北到南 (直行)
                'route_ew': 'e_t t_w', # 东到西 (直行)
                'route_sn': 's_t t_n', # 南到北 (直行)
                'route_we': 'w_t t_e', # 西到东 (直行)
                'route_ne': 'n_t t_e', # 北到东 (左转)
                'route_es': 'e_t t_s', # 东到南 (左转)
                'route_sw': 's_t t_w', # 南到西 (左转)
                'route_wn': 'w_t t_n'  # 西到北 (左转)
            }
    else:
        # 使用默认示例路线
        example_routes = {
            'route_ns': 'n_t t_s', # 北到南 (直行)
            'route_ew': 'e_t t_w', # 东到西 (直行)
            'route_sn': 's_t t_n', # 南到北 (直行)
            'route_we': 'w_t t_e', # 西到东 (直行)
            'route_ne': 'n_t t_e', # 北到东 (左转)
            'route_es': 'e_t t_s', # 东到南 (左转)
            'route_sw': 's_t t_w', # 南到西 (左转)
            'route_wn': 'w_t t_n'  # 西到北 (左转)
        }

    for route_id, edges in example_routes.items():
        ET.SubElement(routes_element, 'route', {'id': route_id, 'edges': edges})

    # 3. 生成流量 (flow)
    flow_common_params = {
        'departSpeed': 'max',
        'departPos': 'base',
        'departLane': 'best'
    }

    # 基础流量值 - 根据提取的路线动态生成
    base_flow_values = {}
    
    if flow_type == 'perhour':
        attr_name = 'vehsPerHour'
        # 为每条路线设置基础流量
        for route_id in example_routes.keys():
            if 'we' in route_id.lower() or 'ew' in route_id.lower():
                base_flow_values[route_id] = 300  # 东西向主干道
            elif 'ns' in route_id.lower() or 'sn' in route_id.lower():
                base_flow_values[route_id] = 300  # 南北向主干道
            else:
                base_flow_values[route_id] = 60   # 左转等其他路线
                
    elif flow_type == 'probability':
        attr_name = 'probability'
        # 为每条路线设置基础概率
        for route_id in example_routes.keys():
            if 'we' in route_id.lower() or 'ew' in route_id.lower():
                base_flow_values[route_id] = 0.3  # 东西向主干道
            elif 'ns' in route_id.lower() or 'sn' in route_id.lower():
                base_flow_values[route_id] = 0.4  # 南北向主干道
            else:
                base_flow_values[route_id] = 0.1  # 左转等其他路线
    else:
        raise ValueError("flow_type 必须是 'perhour' 或 'probability'")

    # 如果没有提供时间剖面，则默认为整个时长内的恒定流量
    if time_profiles is None:
        time_profiles = [(0, total_duration, 1.0)] # (开始时间, 结束时间, 流量乘数)

    # 为每个时间剖面段生成流量
    for start_time, end_time, multiplier in time_profiles:
        for route_id, edges in example_routes.items():
            flow_id_suffix = route_id.replace('route_', '')
            # 为每个时间段和路线生成唯一的 flow ID
            flow_id = f"flow_{flow_id_suffix}_{start_time}_{end_time}"

            base_value = base_flow_values[route_id]
            
            # 应用时间剖面乘数
            current_value = base_value * multiplier

            # 引入随机波动
            if random_factor > 0:
                variation_amount = current_value * random_factor
                current_value = random.uniform(current_value - variation_amount, current_value + variation_amount)
                
                # 确保概率值在 [0, 1] 之间
                if flow_type == 'probability':
                    current_value = max(0.0, min(1.0, current_value))
                # 确保 vehsPerHour 是非负整数
                elif flow_type == 'perhour':
                    current_value = max(0, int(current_value))

            ET.SubElement(routes_element, 'flow', {
                'id': flow_id,
                'route': route_id,
                'begin': str(start_time),
                'end': str(end_time),
                **flow_common_params,
                attr_name: str(current_value)
            })

    # 4. 美化XML并保存到文件
    rough_string = ET.tostring(routes_element, 'utf-8')
    reparsed_string = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed_string.toprettyxml(indent="    ")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)

    print(f"成功生成文件: {output_filename}")

# --- 示例用法 ---
if __name__ == "__main__":
    # 示例 1: 恒定流量，但引入 +/-10% 的随机波动
    generate_rou_file(
        'dynamic_perhour_random.rou.xml',
        flow_type='perhour',
        total_duration=3600,
        random_factor=0.1 # +/- 10% 随机波动
    )

    # 示例 2: 时间变化流量 (模拟高峰/平峰)，并引入少量随机波动
    # 模拟一个 2 小时（7200 秒）的仿真
    # 第一个小时 (0-3600s): 正常流量 (乘数 1.0)
    # 第二个小时 (3600-7200s): 高峰流量 (乘数 1.5)
    time_profiles_example = [
        (0, 3600, 1.0),
        (3600, 7200, 1.5)
    ]
    generate_rou_file(
        'dynamic_perhour_time_varying.rou.xml',
        flow_type='perhour',
        total_duration=7200,
        time_profiles=time_profiles_example,
        random_factor=0.05 # 每个时间段内 +/- 5% 的随机波动
    )

    # 示例 3: 基于概率的流量，时间变化且有随机性
    time_profiles_prob_example = [
        (0, 1800, 0.8),   # 前 30 分钟: 较低流量
        (1800, 3600, 1.2), # 后 30 分钟: 较高流量
        (3600, 5400, 0.9)  # 再 30 分钟: 略低
    ]
    generate_rou_file(
        'dynamic_probability_time_varying.rou.xml',
        flow_type='probability',
        total_duration=5400,
        time_profiles=time_profiles_prob_example,
        random_factor=0.08
    )

    print("\n请检查当前目录下的 'dynamic_perhour_random.rou.xml', 'dynamic_perhour_time_varying.rou.xml', 和 'dynamic_probability_time_varying.rou.xml' 文件。")
    print("这些文件现在包含了时间变化和随机性。")
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

def extract_routes_from_template(template_file: str) -> Dict[str, str]:
    """
    从模板路线文件中提取路线ID和对应的边定义
    
    Args:
        template_file (str): 模板路线文件路径
        
    Returns:
        Dict[str, str]: 路线ID到边定义的映射
    """
    try:
        tree = ET.parse(template_file)
        root = tree.getroot()
        
        routes = {}
        for route in root.findall('route'):
            route_id = route.get('id')
            edges = route.get('edges')
            if route_id and edges:
                routes[route_id] = edges
                
        if not routes:
            raise ValueError("模板文件中未找到任何路线定义")
            
        return routes
        
    except FileNotFoundError:
        raise FileNotFoundError(f"模板文件不存在: {template_file}")
    except ET.ParseError as e:
        raise ValueError(f"XML解析错误: {e}")
    except Exception as e:
        raise RuntimeError(f"提取路线时发生错误: {e}")

def generate_curriculum_flow(
    base_route_file: str,
    output_file: str,
    total_sim_seconds: int,
    stage_definitions: List[Dict[str, Any]],
    route_distribution: Dict[str, float]
):
    """
    Generates a SUMO route file with staged traffic flows based on configuration.

    Args:
        base_route_file (str): Path to a .rou.xml file containing only <route> definitions.
        output_file (str): Path where the generated .rou.xml file will be saved.
        total_sim_seconds (int): The total duration of the static flow phases.
        stage_definitions (List[Dict[str, Any]]): A list of dictionaries, each defining a stage.
            Example: [
                {'name': 'low', 'duration_ratio': 0.3, 'flow_rate_multiplier': 0.5},
                {'name': 'medium', 'duration_ratio': 0.4, 'flow_rate_multiplier': 1.0},
                {'name': 'high', 'duration_ratio': 0.3, 'flow_rate_multiplier': 2.0}
            ]
        route_distribution (Dict[str, float]): A dictionary mapping route IDs to their base flow rate (in vehsPerHour).
            Example: {'route_WE': 200, 'route_NS': 150}
    """
    # 1. 解析基础路线文件
    tree = ET.parse(base_route_file)
    root = tree.getroot()

    # 2. 根据配置计算并添加 <flow>
    current_begin_time = 0
    for stage in stage_definitions:
        duration = int(total_sim_seconds * stage['duration_ratio'])
        end_time = current_begin_time + duration
        flow_multiplier = stage['flow_rate_multiplier']

        for route_id, base_rate in route_distribution.items():
            flow_rate = int(base_rate * flow_multiplier)
            
            flow_attrib = {
                'id': f"{stage['name']}_{route_id}",
                'type': 'DEFAULT_VEHTYPE',
                'route': route_id,
                'begin': str(current_begin_time),
                'end': str(end_time),
                'vehsPerHour': str(flow_rate)
            }
            flow_element = ET.Element('flow', attrib=flow_attrib)
            root.append(flow_element)

        current_begin_time = end_time

    # 3. 写入到输出文件
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_file)
    
    # 返回静态流量阶段的总时长，用于SumoEnv的dynamic_start_time
    return current_begin_time

import osmnx as ox
import networkx as nx
import math
from dotenv import load_dotenv
import os


def get_graph_stats(bbox):
    """지정된 경계 내 노드와 간선 수를 계산합니다."""
    south = bbox['south']
    west = bbox['west']
    north = bbox['north']
    east = bbox['east']
    G = ox.graph_from_bbox((west, south, east, north), network_type='all', simplify=True)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    
def find_circuits_in_area(bbox, progress_callback=None):
    """경계 내에서 서킷을 찾아 점수를 매깁니다."""
    # .env 파일에서 환경 변수 불러오기
    load_dotenv()
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    south = bbox['south']
    west = bbox['west']
    north = bbox['north']
    east = bbox['east']
    
    if progress_callback:
        progress_callback(10, "도로망 데이터 불러오는 중...")
    
    G = ox.graph_from_bbox((west, south, east, north), network_type='all', simplify=True)
    
    if progress_callback:
        progress_callback(30, "고도 데이터 추가 중...")

    G = ox.elevation.add_node_elevations_google(G, api_key=GOOGLE_MAPS_API_KEY)
    G = ox.elevation.add_edge_grades(G)

    circuits = []
    min_length_meters = 3500
    max_length_meters = 7000

    if progress_callback:
        progress_callback(50, "서킷 후보를 탐색 중...")
    
    # 순환 경로 탐색 (간단한 예시)
    for path in nx.simple_cycles(G):
        length = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            # G.get_edge_data(u, v, key=0)는 (u, v) 간선의 첫 번째 속성을 가져옵니다.
            # 'length' 속성이 없으면 기본값으로 0을 사용합니다.
            edge_data = G.get_edge_data(u, v, key=0)
            if edge_data and 'length' in edge_data:
                length += edge_data['length']

        if min_length_meters <= length <= max_length_meters:
            circuits.append(path)

    if progress_callback:
        progress_callback(70, f"총 {len(circuits)}개의 후보 서킷 평가 중...")

    # 서킷 평가 및 점수화
    scored_circuits = []
    for path in circuits:
        num_corners = calculate_corners(G, path)
        elevation_change = calculate_elevation_change(G, path)
        straight_length = find_longest_straight_segment(G, path)
        total_length = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            edge_data = G.get_edge_data(u, v, key=0)
            if edge_data and 'length' in edge_data:
                total_length += edge_data['length']
        total_length = calculate_total_length(G, path)

        score = (num_corners * 0.4) + (elevation_change * 0.3) + (straight_length * 0.3)
        
        scored_circuits.append({
            "path": path,
            "score": score,
            "length": total_length,
            "corners": num_corners,
            "elevation_change": elevation_change,
            "drs_straight": straight_length
        })

    scored_circuits.sort(key=lambda x: x['score'], reverse=True)
    
    for circuit in scored_circuits:
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in circuit['path']]
        circuit['coordinates'] = coords
        del circuit['path']

    return scored_circuits[:5]

def calculate_total_length(G, path):
    """
    Calculate the total length of a path by summing the lengths of its edges.
    """
    total_length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v, key=0)  # Specify key=0 to get the first edge if there are parallel edges
        if edge_data and 'length' in edge_data[0]:
            total_length += edge_data[0]['length']
    return total_length


# --- 헬퍼 함수들 (직접 구현 필요) ---
# 이 함수들의 로직을 직접 구현해야 합니다.
def calculate_corners(G, path):
    """
    주어진 서킷 경로에서 코너의 수를 계산합니다.
    임계값보다 작은 각도를 가진 회전을 코너로 정의합니다.
    """
    if len(path) < 3:
        return 0

    num_corners = 0
    angle_threshold = 160  # 160도보다 작은 각도는 코너로 간주합니다.

    # 경로의 모든 노드를 순회하며 연속된 세 노드로 각도를 계산합니다.
    for i in range(1, len(path) - 1):
        # 연속된 세 노드의 좌표를 가져옵니다. (n1 -> n2 -> n3)
        n1 = path[i-1]
        n2 = path[i]
        n3 = path[i+1]
        
        coords1 = (G.nodes[n1]['x'], G.nodes[n1]['y'])
        coords2 = (G.nodes[n2]['x'], G.nodes[n2]['y'])
        coords3 = (G.nodes[n3]['x'], G.nodes[n3]['y'])

        # 두 도로 구간의 벡터를 계산합니다.
        v1 = (coords2[0] - coords1[0], coords2[1] - coords1[1])
        v2 = (coords3[0] - coords2[0], coords3[1] - coords2[1])

        # 내적과 크기(magnitude)를 계산합니다.
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        # 0으로 나누는 것을 방지합니다.
        if mag1 == 0 or mag2 == 0:
            continue

        # 각도를 도(degree) 단위로 계산합니다.
        angle = math.degrees(math.acos(min(max(dot_product / (mag1 * mag2), -1.0), 1.0)))
        
        # 각도가 임계값보다 작으면 코너로 간주합니다.
        if angle < angle_threshold:
            num_corners += 1

    return num_corners

def calculate_elevation_change(G, path):
    """
    경로를 따라 총 절대 고도 변화(오르막 + 내리막)를 계산합니다.
    """
    total_change = 0
    # 경로의 모든 구간을 순회합니다.
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # 구간의 시작점과 끝점의 고도를 가져옵니다.
        # 고도 데이터가 없는 경우를 대비해 예외 처리가 필요할 수 있습니다.
        elev1 = G.nodes[u].get('elevation_m', 0)
        elev2 = G.nodes[v].get('elevation_m', 0)
        
        # 절대값 차이를 모두 더하여 총 변화량을 구합니다.
        total_change += abs(elev2 - elev1)
        
    return total_change

def find_longest_straight_segment(G, path):
    """
    서킷에서 가장 긴 직선 구간의 길이를 찾습니다.
    '직선'은 작은 회전만 있는 연속적인 도로 구간입니다.
    """
    if len(path) < 2:
        return 0

    max_straight_length = 0
    current_straight_length = 0
    straight_nodes_segment = []
    straight_angle_threshold = 175 # 175도보다 작은 각도는 '직선'이 끝났음을 의미합니다.

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        # 현재 구간의 길이를 더합니다.
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            continue
        current_straight_length += edge_data[0]['length']

        # 만약 다음 구간이 존재한다면 각도를 계산합니다.
        if i + 2 < len(path):
            n1 = path[i]
            n2 = path[i+1]
            n3 = path[i+2]
            
            coords1 = (G.nodes[n1]['x'], G.nodes[n1]['y'])
            coords2 = (G.nodes[n2]['x'], G.nodes[n2]['y'])
            coords3 = (G.nodes[n3]['x'], G.nodes[n3]['y'])
            
            v1 = (coords2[0] - coords1[0], coords2[1] - coords1[1])
            v2 = (coords3[0] - coords2[0], coords3[1] - coords2[1])

            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 == 0 or mag2 == 0:
                continue
            
            angle = math.degrees(math.acos(min(max(dot_product / (mag1 * mag2), -1.0), 1.0)))

            # 회전 각도가 크면, 직선 구간이 종료된 것으로 간주합니다.
            if angle < straight_angle_threshold:
                max_straight_length = max(max_straight_length, current_straight_length)
                current_straight_length = 0 # 직선 길이 리셋
        else:
            # 경로의 마지막 구간입니다. 현재 길이를 최대값과 비교합니다.
            max_straight_length = max(max_straight_length, current_straight_length)

    return max_straight_length
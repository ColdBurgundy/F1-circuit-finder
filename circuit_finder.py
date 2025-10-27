import osmnx as ox
import networkx as nx
import math
from dotenv import load_dotenv
import random
import hashlib
import os
from datetime import datetime

# --- 캐시 설정 ---
CACHE_DIR = "graph_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- 기존 함수들 ---
def get_graph_stats(bbox):
    """지정된 경계 내 노드와 간선 수를 계산합니다."""
    south = bbox['south']
    west = bbox['west']
    north = bbox['north']
    east = bbox['east']
    G = ox.graph_from_bbox((west, south, east, north), network_type='all', simplify=True)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}
    
def get_or_create_filtered_graph(bbox, progress_callback=None, check_grade=False):
    """
    캐시된 필터링 그래프를 로드하거나, 없을 경우 새로 생성하고 캐시합니다.
    필터링 조건: 4차선 이상, 경사도 10% 미만
    """
    # bbox를 기반으로 고유한 파일 이름 생성
    bbox_str = f"{bbox['north']:.6f},{bbox['south']:.6f},{bbox['east']:.6f},{bbox['west']:.6f},grade:{check_grade}"
    hash_prefix = f"graph_{hashlib.md5(bbox_str.encode()).hexdigest()}"
    
    # 해시 접두사로 시작하는 기존 캐시 파일 검색
    try:
        existing_files = [f for f in os.listdir(CACHE_DIR) if f.startswith(hash_prefix) and f.endswith('.graphml')]
        if existing_files:
            filepath = os.path.join(CACHE_DIR, existing_files[0])
            if progress_callback: progress_callback(40, "캐시된 도로망 데이터 로드 중...")
            G_loaded = ox.load_graphml(filepath)
            nodes, edges = ox.graph_to_gdfs(G_loaded)
            G = ox.graph_from_gdfs(nodes, edges, graph_attrs=G_loaded.graph)
            return G
    except FileNotFoundError:
        pass # 캐시 디렉토리가 아직 없으면 그냥 진행

    # 캐시 파일이 없으면 새로 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(CACHE_DIR, f"{hash_prefix}_{timestamp}.graphml")

    # .env 파일에서 환경 변수 불러오기
    load_dotenv()
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
    south = bbox['south']
    west = bbox['west']
    north = bbox['north']
    east = bbox['east']
    
    if progress_callback: progress_callback(10, "도로망 데이터 불러오는 중...")
    G = ox.graph_from_bbox((west, south, east, north), network_type='all', simplify=True)
    
    if progress_callback: progress_callback(30, "고도 데이터 추가 중...")
    G = ox.elevation.add_node_elevations_google(G, api_key=GOOGLE_MAPS_API_KEY)
    G = ox.elevation.add_edge_grades(G)

    if progress_callback: progress_callback(40, "도로망 필터링 및 캐시 저장 중...")
    
    # 조건에 맞는 간선만 필터링
    valid_edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        # 필터링 1: 경사도 10% 초과 시 제외
        if check_grade and abs(data.get('grade', 0)) > 0.1:
            continue

        # 필터링 2: 도로가 4차선 미만이면 제외
        lanes = data.get('lanes')
        width = data.get('width')
        if isinstance(lanes, list):
            # 'lanes'가 리스트인 경우, 숫자 값의 합계를 사용
            try: lanes = sum(int(l) for l in lanes)
            except (ValueError, TypeError): lanes = 0
        if isinstance(width, list):
            # 'width'가 리스트인 경우, 첫 번째 유효한 숫자 값을 사용
            try: width = float(width[0])
            except (ValueError, IndexError): width = 0
        
        is_wide_enough = (lanes and int(lanes) >= 4) or (width and float(width) >= 10)
        if not is_wide_enough: continue
        
        valid_edges.append((u, v, key))

    # 유효한 간선과 그에 연결된 노드로 새로운 그래프 생성
    G_filtered = G.edge_subgraph(valid_edges).copy()

    ox.save_graphml(G_filtered, filepath=filepath)
    
    return G_filtered

def find_circuits_in_area(bbox, progress_callback=None, check_curvature=False, check_grade=False):
    """경계 내에서 서킷을 찾아 점수를 매깁니다."""
    G = get_or_create_filtered_graph(bbox, progress_callback, check_grade=check_grade)

    circuits = []
    min_length_meters = 3200  # 최소 트랙 길이 3.2km
    max_length_meters = 7000  # 최대 트랙 길이 7km

    if progress_callback:
        progress_callback(50, "서킷 후보를 탐색 중...")
    
    # 필터링된 그래프에서 사이클 탐색
    nodes = list(G.nodes())
    if not nodes: return [] # 필터링 후 도로가 없으면 빈 리스트 반환
    start_nodes = random.sample(nodes, min(len(nodes), 200)) # 샘플링하여 시작
    
    # DFS를 통해 길이 조건에 맞는 기본 후보군 탐색
    candidate_cycles = find_cycles_with_dfs(G, start_nodes, min_length_meters, max_length_meters)

    # 상세 필터링 적용
    for path in candidate_cycles:
        if is_path_valid(G, path, check_curvature=check_curvature):
            circuits.append(path)
    
    if progress_callback:
        progress_callback(70, f"총 {len(circuits)}개의 후보 서킷 평가 중...")

    # 서킷 평가 및 점수화
    scored_circuits = []
    for path in circuits:
        num_corners = calculate_corners(G, path)
        elevation_change = calculate_elevation_change(G, path)
        straight_length = find_longest_straight_segment(G, path)
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

def is_path_valid(G, path, check_curvature=False):
    """
    탐색된 경로가 F1 서킷의 상세 규정(직선, 너비, 곡률, 코너 거리)을 만족하는지 검증합니다.
    """
    # 1. 고속 직선 구간: 최소 800m 이상의 직선 구간 1개 이상 포함
    longest_straight = find_longest_straight_segment(G, path)
    if longest_straight < 800:
        return False

    # 2. 최소 곡률 반경 및 코너 간 최소 거리 검증
    corner_nodes = calculate_corners(G, path, return_nodes=True)
    if not corner_nodes: # 코너가 없으면 서킷이 아님
        return False

    path_with_loop = path + [path[0]]
    node_to_dist = {node: 0 for node in path}
    dist = 0
    for i in range(len(path)):
        u, v = path_with_loop[i], path_with_loop[i+1]
        dist += G.get_edge_data(u, v, key=0).get('length', 0)
        # node_to_dist에 v가 없는 경우를 대비하여 초기화
        if v not in node_to_dist: node_to_dist[v] = 0
        node_to_dist[v] = dist

    corner_distances = sorted([node_to_dist[cn] for cn in corner_nodes])

    # 코너 간 거리 확인 (순환 구조 고려)
    for i in range(len(corner_distances) - 1):
        if corner_distances[i+1] - corner_distances[i] < 100:
            return False
    total_length = calculate_total_length(G, path)
    # 마지막 코너와 첫 코너 사이의 거리
    if (total_length - corner_distances[-1] + corner_distances[0]) < 100:
        return False

    # 최소 곡률 반경 확인
    if check_curvature:
        for corner_node in corner_nodes:
            idx = path.index(corner_node)
            p_prev = path[idx-1]
            p_curr = corner_node
            p_next = path[(idx+1) % len(path)]
            
            radius = calculate_curvature_radius(G.nodes[p_prev], G.nodes[p_curr], G.nodes[p_next])
            if radius < 30:
                return False

    return True

def find_cycles_with_dfs(G, start_nodes, min_len, max_len):
    """
    조기 필터링이 적용된 DFS를 사용하여 지정된 길이 범위 내의 사이클을 찾습니다.
    """
    cycles = []
    found_cycles_sorted = set() # 중복 경로 확인용

    for start_node in start_nodes:
        stack = [(start_node, [start_node], 0)]
        
        while stack:
            current_node, path, current_length = stack.pop()

            # G[current_node]는 이웃 노드와 그 사이의 모든 간선 정보를 담고 있습니다.
            for neighbor, edges in G[current_node].items():
                if neighbor in path and neighbor != start_node:
                    continue

                # 두 노드 사이에 여러 간선(key=0, key=1...)이 있을 수 있으므로 모두 순회합니다.
                for key in edges:
                    edge_data = G.get_edge_data(current_node, neighbor, key=key)
                    if not (edge_data and 'length' in edge_data): continue

                    new_length = current_length + edge_data['length']
                    if new_length > max_len: continue

                    if neighbor == start_node:
                        if min_len <= new_length <= max_len:
                            sorted_path_tuple = tuple(sorted(path))
                            if sorted_path_tuple not in found_cycles_sorted:
                                cycles.append(path)
                                found_cycles_sorted.add(sorted_path_tuple)
                    elif neighbor not in path:
                        stack.append((neighbor, path + [neighbor], new_length))
    return cycles

def calculate_total_length(G, path):
    """
    Calculate the total length of a path by summing the lengths of its edges.
    """
    total_length = 0
    extended_path = path + [path[0]]
    for i in range(len(extended_path) - 1):
        u, v = extended_path[i], extended_path[i + 1]
        edge_data = G.get_edge_data(u, v, key=0)
        if edge_data and 'length' in edge_data:
            total_length += edge_data['length']
    return total_length

def haversine_distance(lat1, lon1, lat2, lon2):
    """ 두 GPS 좌표 간의 거리를 미터 단위로 계산 """
    R = 6371e3  # 지구 반지름 (미터)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_curvature_radius(p1, p2, p3):
    """ 세 점(노드)으로 정의된 코너의 곡률 반경을 계산합니다. """
    # 좌표 추출
    x1, y1 = p1['x'], p1['y']
    x2, y2 = p2['x'], p2['y']
    x3, y3 = p3['x'], p3['y']

    # 세 점을 지나는 원(외접원)의 반지름 계산
    # 수식 참조: https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: # 세 점이 거의 일직선상에 있는 경우
        return float('inf')

    # 각 변의 길이의 제곱
    a2 = (x2 - x3)**2 + (y2 - y3)**2
    b2 = (x1 - x3)**2 + (y1 - y3)**2
    c2 = (x1 - x2)**2 + (y1 - y2)**2

    # 실제 거리(미터)로 변환하기 위한 근사치 계산 (위도 37도 기준)
    # 1도 = 약 111km
    m_per_deg_lat = 111000
    m_per_deg_lon = 111000 * math.cos(math.radians(p2['y']))
    
    a = math.sqrt(a2) * m_per_deg_lon # x축 변화가 더 크다고 가정
    b = math.sqrt(b2) * m_per_deg_lon
    c = math.sqrt(c2) * m_per_deg_lon

    # 외접원 반지름 공식: R = abc / sqrt((a+b+c)(a+b-c)(a-b+c)(-a+b+c))
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return float('inf')
        
    radius = (a * b * c) / (4.0 * math.sqrt(area_sq))
    return radius


# --- 헬퍼 함수들 (직접 구현 필요) ---
# 이 함수들의 로직을 직접 구현해야 합니다.
def calculate_corners(G, path, return_nodes=False):
    """
    주어진 서킷 경로에서 코너의 수를 계산합니다.
    임계값보다 작은 각도를 가진 회전을 코너로 정의합니다.
    return_nodes=True이면 코너 노드 리스트를 반환합니다.
    """
    if len(path) < 3:
        return [] if return_nodes else 0

    corners = []
    angle_threshold = 160  # 이 각도보다 작으면 코너로 간주

    extended_path = path + path[:2] # 순환 및 마지막 코너 계산을 위해 2개 노드 추가

    for i in range(len(path)):
        n1 = extended_path[i]
        n2 = extended_path[i+1]
        n3 = extended_path[i+2]
        
        coords1 = (G.nodes[n1]['x'], G.nodes[n1]['y'])
        coords2 = (G.nodes[n2]['x'], G.nodes[n2]['y'])
        coords3 = (G.nodes[n3]['x'], G.nodes[n3]['y'])

        v1 = (coords2[0] - coords1[0], coords2[1] - coords1[1])
        v2 = (coords3[0] - coords2[0], coords3[1] - coords2[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0: continue

        angle = math.degrees(math.acos(min(max(dot_product / (mag1 * mag2), -1.0), 1.0)))
        
        if angle < angle_threshold:
            corners.append(n2) # 코너는 가운데 노드

    return corners if return_nodes else len(corners)

def calculate_elevation_change(G, path):
    """
    경로를 따라 총 절대 고도 변화(오르막 + 내리막)를 계산합니다.
    """
    total_change = 0
    extended_path = path + [path[0]]
    for i in range(len(extended_path) - 1):
        u, v = extended_path[i], extended_path[i+1]
        
        elev1 = G.nodes[u].get('elevation', 0)
        elev2 = G.nodes[v].get('elevation', 0)
        
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
    straight_angle_threshold = 175

    doubled_path = path + path

    for i in range(len(doubled_path) - 1):
        u, v = doubled_path[i], doubled_path[i+1]

        edge_data = G.get_edge_data(u, v, key=0)
        if not (edge_data and 'length' in edge_data):
            current_straight_length = 0 # 도로 정보 없으면 직선 종료
            continue
        current_straight_length += edge_data['length']

        if i >= len(doubled_path) - 2:
            break

        p1, p2, p3 = doubled_path[i], doubled_path[i+1], doubled_path[i+2]
        coords1 = (G.nodes[p1]['x'], G.nodes[p1]['y'])
        coords2 = (G.nodes[p2]['x'], G.nodes[p2]['y'])
        coords3 = (G.nodes[p3]['x'], G.nodes[p3]['y'])

        v1 = (coords2[0] - coords1[0], coords2[1] - coords1[1])
        v2 = (coords3[0] - coords2[0], coords3[1] - coords2[1])

        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            angle = 180.0
        else:
            angle = math.degrees(math.acos(min(max(dot_product / (mag1 * mag2), -1.0), 1.0)))

        if angle < straight_angle_threshold:
            max_straight_length = max(max_straight_length, current_straight_length)
            current_straight_length = 0

    max_straight_length = max(max_straight_length, current_straight_length)

    return max_straight_length
import osmnx as ox
import networkx as nx
import sys

# Setting a very small, specific bounding box in Seoul (around Gwanghwamun Square).
# This area is small enough to be downloaded without issue.
north, south, east, west = 37.575, 37.574, 126.977, 126.976

print("테스트 시작...")
print("OSMnx 라이브러리를 성공적으로 불러왔습니다.")

try:
    print(f"경계 박스 좌표: 북={north}, 남={south}, 동={east}, 서={west}")
    print("도로 그래프를 다운로드 중입니다. 잠시만 기다려 주세요...")
    
    # Passing the bounding box as a single tuple, which is the correct way
    # to call this function in modern osmnx versions.
    bounding_box = (west, south, east, north)
    G = ox.graph_from_bbox(bounding_box, network_type='all')
    
    print("도로 그래프 다운로드 및 생성이 완료되었습니다.")
    
    # Calculate the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    print("노드와 간선 수 계산이 완료되었습니다.")
    print(f"최종 노드 수: {num_nodes}")
    print(f"최종 간선 수: {num_edges}")
    
    if num_nodes > 0:
        print("성공! osmnx가 데이터를 제대로 가져오고 있습니다.")
    else:
        print("실패: 그래프는 생성되었지만 노드가 없습니다. 다른 지역을 시도해 보세요.")

except Exception as e:
    print("-" * 30)
    print("오류 발생: 스크립트 실행 중 예외가 발생했습니다.")
    print(f"오류 메시지: {e}")
    import traceback
    traceback.print_exc()
    print("-" * 30)
finally:
    print("테스트 스크립트 종료.")
    sys.stdout.flush()
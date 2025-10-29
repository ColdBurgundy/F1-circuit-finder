from flask import Flask, render_template, request, jsonify
from threading import Thread, Timer
import uuid

# circuit_finder.py에서 함수를 가져옵니다.
from circuit_finder import find_circuits_in_area, get_graph_stats, get_or_create_filtered_graph

app = Flask(__name__)

# 진행 상태를 저장할 전역 변수
# 각 요청(task)별 진행 상태를 저장하는 딕셔너리
# { "task_id_1": {"progress": 0, "message": "..."}, "task_id_2": ... }
tasks = {}

def update_progress(task_id, progress, message, circuits=None):
    """특정 작업 ID에 대한 진행 상태를 업데이트합니다."""
    tasks[task_id] = {"progress": progress, "message": message}
    if circuits is not None:
        tasks[task_id]["circuits"] = circuits

def cleanup_task(task_id):
    """지정된 시간 후 완료된 작업을 딕셔너리에서 제거합니다."""
    if task_id in tasks:
        del tasks[task_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prepare-graph', methods=['POST'])
def prepare_graph():
    """도로망 데이터를 다운로드, 필터링, 캐싱하는 작업을 시작합니다."""
    data = request.get_json()
    bbox = data.get('bbox')
    if not bbox:
        return jsonify({"error": "경계 상자가 제공되지 않았습니다."}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"progress": 0, "message": "데이터 준비 대기 중..."}

    progress_callback = lambda progress, message: update_progress(task_id, progress, message)
    
    # get_or_create_filtered_graph 함수만 스레드에서 실행
    # 이 함수는 데이터가 캐시되어 있지 않으면 생성하고, 있으면 그냥 반환합니다.
    thread = Thread(target=run_prepare_graph_in_thread, args=(task_id, bbox, progress_callback))
    thread.start()

    return jsonify({"status": "processing", "task_id": task_id})

@app.route('/api/find-circuits', methods=['POST'])
def find_circuits():
    data = request.get_json()
    bbox = data.get('bbox')
    # 프론트엔드에서 서킷 크기 선택 값을 받습니다. (기본값: F1 규격)
    min_length = data.get('min_length', 3200)
    max_length = data.get('max_length', 7000)

    # 이 줄을 추가하여 서버가 받은 데이터를 확인합니다.
    print('서버가 받은 경계 박스:', bbox)
    print(f'서버가 받은 서킷 크기: {min_length}m ~ {max_length}m')
    
    if not bbox:
        return jsonify({"error": "경계 상자가 제공되지 않았습니다."}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"progress": 0, "message": "탐색 대기 중..."}

    # 서킷 탐색을 별도의 스레드에서 실행
    # progress_callback에 task_id를 넘겨주기 위해 lambda 함수 사용
    progress_callback = lambda progress, message: update_progress(task_id, progress, message)
    thread = Thread(target=run_search_in_thread, args=(task_id, bbox, min_length, max_length, progress_callback))
    thread.start()
    
    return jsonify({"status": "processing", "task_id": task_id})

def run_search_in_thread(task_id, bbox, min_length, max_length, progress_callback): # find_circuits를 위한 스레드 함수
    """스레드에서 서킷 탐색을 실행하고 상태를 업데이트합니다."""
    try:
        # check_grade=False로 설정하여 경사도 필터링을 명시적으로 비활성화합니다.
        circuits = find_circuits_in_area(
            bbox, 
            min_length_meters=min_length,
            max_length_meters=max_length,
            progress_callback=progress_callback, 
            check_grade=False
        )
        
        if not circuits:
            update_progress(task_id, 100, "해당 지역에는 서킷을 생성할 수 없습니다.")
        else:
            update_progress(task_id, 100, "완료", circuits=circuits)

    except Exception as e:
        # 개발자를 위해 터미널에 전체 오류 로그 출력
        app.logger.error(f"Task {task_id} failed:", exc_info=e)
        # 사용자에게는 간결한 오류 메시지 전달
        update_progress(task_id, -1, f"오류 발생: {str(e)}")
    finally:
        # 10분(600초) 후에 이 작업에 대한 정보를 메모리에서 삭제
        Timer(600, cleanup_task, args=[task_id]).start()

def run_prepare_graph_in_thread(task_id, bbox, progress_callback): # prepare_graph를 위한 스레드 함수
    """스레드에서 그래프 데이터 준비 및 캐싱을 실행합니다."""
    try:
        # 이 함수는 캐시가 없으면 생성하고, 있으면 빠르게 로드만 합니다.
        # check_grade=False는 find_circuits_in_area 호출과 일관성을 맞추기 위함입니다.
        get_or_create_filtered_graph(bbox, progress_callback=progress_callback, check_grade=False)
        update_progress(task_id, 100, "지도 데이터 준비 완료")
    except Exception as e:
        app.logger.error(f"Task {task_id} failed:", exc_info=e)
        update_progress(task_id, -1, f"오류 발생: {str(e)}")
    finally:
        # 10분(600초) 후에 이 작업에 대한 정보를 메모리에서 삭제
        Timer(600, cleanup_task, args=[task_id]).start()


@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """특정 작업 ID에 대한 진행 상태를 반환합니다."""
    return jsonify(tasks.get(task_id, {"progress": -1, "message": "알 수 없는 작업 ID입니다."}))

@app.route('/api/graph-stats', methods=['POST'])
def get_stats():
    data = request.get_json()
    bbox = data.get('bbox')
    
    if not bbox:
        return jsonify({"error": "경계 상자가 제공되지 않았습니다."}), 400
    
    try:
        stats = get_graph_stats(bbox)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
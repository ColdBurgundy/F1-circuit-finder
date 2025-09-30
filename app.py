from flask import Flask, render_template, request, jsonify
from threading import Thread
import time

# circuit_finder.py에서 함수를 가져옵니다.
from circuit_finder import find_circuits_in_area, get_graph_stats

app = Flask(__name__)

# 진행 상태를 저장할 전역 변수
# 멀티스레딩 환경에서 전역 변수를 사용할 때 lock을 사용해야 하지만,
# 여기서는 간단한 예시를 위해 생략합니다.
progress_status = {"progress": 0, "message": "대기 중..."}

def update_progress(progress, message):
    global progress_status
    progress_status = {"progress": progress, "message": message}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/find-circuits', methods=['POST'])
def find_circuits():
    data = request.get_json()
    bbox = data.get('bbox')

    # 이 줄을 추가하여 서버가 받은 데이터를 확인합니다.
    print('서버가 받은 경계 박스:', bbox)
    
    if not bbox:
        return jsonify({"error": "경계 상자가 제공되지 않았습니다."}), 400

    # 서킷 탐색을 별도의 스레드에서 실행
    thread = Thread(target=run_search_in_thread, args=(bbox,))
    thread.start()
    
    return jsonify({"status": "processing"})

def run_search_in_thread(bbox):
    global progress_status
    progress_status = {"progress": 0, "message": "도로 데이터를 불러오는 중..."}
    
    try:
        circuits = find_circuits_in_area(bbox, progress_callback=update_progress)
        
        if not circuits:
            progress_status = {"progress": 100, "message": "해당 지역에는 서킷을 생성할 수 없습니다."}
        else:
            progress_status = {"progress": 100, "message": "완료", "circuits": circuits}

    except Exception as e:
        progress_status = {"progress": -1, "message": f"오류 발생: {str(e)}"}

@app.route('/api/progress', methods=['GET'])
def get_progress():
    global progress_status
    return jsonify(progress_status)

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
    app.run(debug=True)
// --- 지도 초기화 ---
const map = L.map('map').setView([37.5665, 126.9780], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// --- UI 요소 ---
const analyzeBtn = document.getElementById('analyze-area-btn');
const prepareDataBtn = document.getElementById('prepare-data-btn');
const findCircuitsBtn = document.getElementById('find-circuits-btn');

const statsInfo = document.getElementById('stats-info');
const nodeCountSpan = document.getElementById('node-count');
const edgeCountSpan = document.getElementById('edge-count');

const searchControls = document.getElementById('search-controls');
const circuitSizeSelect = document.getElementById('circuit-size');

const loadingSpinner = document.getElementById('loading-spinner');
const loadingMessage = document.getElementById('loading-message');
const progressBarFill = document.getElementById('progress-bar-fill');

const resultsPanel = document.getElementById('results-panel');
const circuitList = document.getElementById('circuit-list');

let circuitLayers = [];

// --- 이벤트 핸들러 ---
analyzeBtn.addEventListener('click', async () => {
    resetUI();
    setLoading(true, '지역 분석 중...');
    
    const bbox = getBboxFromMap();

    try {
        const response = await fetch('/api/graph-stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bbox })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        nodeCountSpan.textContent = data.nodes;
        edgeCountSpan.textContent = data.edges;
        statsInfo.classList.remove('hidden');
        prepareDataBtn.disabled = false;
    } catch (error) {
        alert('지역 분석 중 오류가 발생했습니다: ' + error.message);
    } finally {
        setLoading(false);
    }
});

prepareDataBtn.addEventListener('click', async () => {
    prepareDataBtn.disabled = true;
    const bbox = getBboxFromMap();
    
    try {
        const response = await fetch('/api/prepare-graph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bbox })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        pollProgress(data.task_id, 'prepare');
    } catch (error) {
        alert('데이터 준비 요청 중 오류가 발생했습니다: ' + error.message);
        prepareDataBtn.disabled = false;
    }
});

findCircuitsBtn.addEventListener('click', async () => {
    findCircuitsBtn.disabled = true;
    const bbox = getBboxFromMap();
    const sizeOption = circuitSizeSelect.value.split(',');
    const minLength = parseInt(sizeOption[0], 10);
    const maxLength = parseInt(sizeOption[1], 10);

    try {
        const response = await fetch('/api/find-circuits', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bbox, min_length: minLength, max_length: maxLength })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        pollProgress(data.task_id, 'search');
    } catch (error) {
        alert('서킷 탐색 요청 중 오류가 발생했습니다: ' + error.message);
        findCircuitsBtn.disabled = false;
    }
});

// --- 헬퍼 함수 ---
function getBboxFromMap() {
    const bounds = map.getBounds();
    return {
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest()
    };
}

function setLoading(isLoading, message = '') {
    if (isLoading) {
        loadingSpinner.classList.remove('hidden');
        loadingMessage.textContent = message;
        progressBarFill.style.width = '0%';
        progressBarFill.textContent = '0%';
    } else {
        loadingSpinner.classList.add('hidden');
    }
}

function resetUI() {
    statsInfo.classList.add('hidden');
    prepareDataBtn.disabled = true;
    prepareDataBtn.textContent = '지도 데이터 준비';
    searchControls.classList.add('hidden');
    resultsPanel.classList.add('hidden');
    circuitList.innerHTML = '';
    circuitLayers.forEach(layer => map.removeLayer(layer));
    circuitLayers = [];
}

function pollProgress(taskId, pollType) {
    setLoading(true, pollType === 'prepare' ? '데이터 준비 중...' : '서킷 탐색 중...');

    const interval = setInterval(async () => {
        try {
            const response = await fetch(`/api/progress/${taskId}`);
            const data = await response.json();

            if (data.progress >= 0) {
                progressBarFill.style.width = data.progress + '%';
                progressBarFill.textContent = data.progress + '%';
                loadingMessage.textContent = data.message;
            }

            if (data.progress >= 100 || data.progress < 0) {
                clearInterval(interval);
                setLoading(false);

                if (data.progress < 0) {
                    alert('오류가 발생했습니다: ' + data.message);
                    return;
                }

                if (pollType === 'prepare') {
                    prepareDataBtn.textContent = '지도 데이터 준비 완료';
                    searchControls.classList.remove('hidden');
                } else if (pollType === 'search') {
                    // [수정] data.circuits가 존재하지 않거나(서킷 못찾음) 빈 배열일 때도 displayResults를 호출하도록 변경
                    displayResults(data.circuits);
                    findCircuitsBtn.disabled = false;
                }
            }
        } catch (error) {
            clearInterval(interval);
            setLoading(false);
            alert('진행 상태를 가져오는 중 오류가 발생했습니다.');
        }
    }, 1000);
}

function displayResults(circuits) {
    if (circuits.length === 0) {
        alert('조건에 맞는 서킷을 찾을 수 없습니다.');
        return;
    }

    resultsPanel.classList.remove('hidden');
    circuits.forEach((circuit, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <h4>서킷 #${index + 1} (점수: ${circuit.score.toFixed(2)})</h4>
            <p>총 길이: ${(circuit.length / 1000).toFixed(2)} km, 코너 수: ${circuit.corners} 개, 최장 직선: ${circuit.drs_straight.toFixed(2)} m</p>
        `;
        li.addEventListener('mouseover', () => {
            circuitLayers[index].setStyle({ color: 'blue', weight: 7 });
        });
        li.addEventListener('mouseout', () => {
            circuitLayers[index].setStyle({ color: 'red', weight: 5 });
        });
        circuitList.appendChild(li);

        const polyline = L.polyline(circuit.coordinates, { color: 'gray', weight: 5, opacity: 0.7 }).addTo(map);
        circuitLayers.push(polyline);
    });
}
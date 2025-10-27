// Initialize the Leaflet map
const map = L.map('map').setView([37.5665, 126.9780], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Get UI elements
const findButton = document.getElementById('find-button');
const checkStatsButton = document.getElementById('check-stats-button');
const resetButton = document.getElementById('reset-button');
const resultsPanel = document.getElementById('results-panel');
const circuitList = document.getElementById('circuit-list');
const loadingSpinner = document.getElementById('loading-spinner');
const nodeCountSpan = document.getElementById('node-count');
const edgeCountSpan = document.getElementById('edge-count');
const loadingMessage = document.getElementById('loading-message');
const progressPercentage = document.getElementById('progress-percentage');

// State variables
let currentCircuitLayers = [];
let progressInterval = null;

// Helper function to get the current map's bounding box
function getBbox() {
    const bounds = map.getBounds();
    return {
        south: bounds.getSouth(),
        west: bounds.getWest(),
        north: bounds.getNorth(),
        east: bounds.getEast()
    };
}

// Helper function to show a loading state
function showLoading(message) {
    loadingSpinner.classList.remove('hidden');
    loadingMessage.textContent = message;
    progressPercentage.textContent = '0';
}

// Helper function to hide the loading state
function hideLoading() {
    loadingSpinner.classList.add('hidden');
}

// Event listener for the "노드/간선 확인" button
checkStatsButton.addEventListener('click', async () => {
    // Check if the zoom level is appropriate
    if (map.getZoom() < 14) {
        alert('더 자세히 확대해야 도로망을 확인할 수 있습니다.');
        return;
    }

    showLoading('노드/간선 수를 계산 중...');
    checkStatsButton.disabled = true;

    try {
        const response = await fetch('/api/graph-stats', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ bbox: getBbox() })
        });
        const data = await response.json();

        if (data.error) {
            alert(`오류: ${data.error}`);
        } else {
            nodeCountSpan.textContent = data.nodes;
            edgeCountSpan.textContent = data.edges;
            // Show the find button and hide the stats button
            checkStatsButton.classList.add('hidden');
            findButton.classList.remove('hidden');
            resetButton.classList.remove('hidden');
        }
    } catch (error) {
        alert('통계 정보를 가져오는 중 오류가 발생했습니다.');
        console.error('Error:', error);
    } finally {
        hideLoading();
        checkStatsButton.disabled = false;
    }
});

// Event listener for the "서킷 찾기" button
findButton.addEventListener('click', async () => {
    showLoading('서킷을 찾는 중...');
    findButton.disabled = true;
    checkStatsButton.disabled = true;
    
    // Start the circuit finding process on the backend
    try {
        const findResponse = await fetch('/api/find-circuits', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ bbox: getBbox() })
        });

        const findData = await findResponse.json();

        if (findData.error) {
            throw new Error(findData.error);
        }

        if (!findData.task_id) {
            throw new Error("서버로부터 작업 ID를 받지 못했습니다.");
        }
        
        // Poll the backend for progress every second
        const taskId = findData.task_id;
        progressInterval = setInterval(async () => {
            try {
                const progressResponse = await fetch(`/api/progress/${taskId}`);
                const progressData = await progressResponse.json();

                loadingMessage.textContent = progressData.message;
                progressPercentage.textContent = progressData.progress;

                if (progressData.progress >= 100 || progressData.progress < 0) {
                    clearInterval(progressInterval);
                    hideLoading();
                    findButton.disabled = false;
                    
                    if (progressData.circuits && progressData.circuits.length > 0) {
                        displayCircuits(progressData.circuits);
                        resultsPanel.classList.remove('hidden');
                    } else {
                        alert(progressData.message || "조건에 맞는 서킷을 찾지 못했습니다.");
                    }
                }
            } catch (pollError) {
                console.error('Polling error:', pollError);
                clearInterval(progressInterval); // Stop polling on error
                alert("진행 상태를 가져오는 중 오류가 발생했습니다.");
            }
        }, 1000);
    } catch (error) {
        alert('서킷을 찾는 중 오류가 발생했습니다.');
        console.error('Error:', error);
        hideLoading();
        findButton.disabled = false;
        if (progressInterval) clearInterval(progressInterval);
    }
});

// Event listener for the "다시하기" button
resetButton.addEventListener('click', () => {
    // Clear all state and UI elements
    currentCircuitLayers.forEach(layer => map.removeLayer(layer));
    currentCircuitLayers = [];
    resultsPanel.classList.add('hidden');
    circuitList.innerHTML = '';
    
    // Reset buttons and stats
    checkStatsButton.classList.remove('hidden');
    findButton.classList.add('hidden');
    resetButton.classList.add('hidden');
    findButton.disabled = false;
    
    nodeCountSpan.textContent = '0';
    edgeCountSpan.textContent = '0';
    
    if (progressInterval) clearInterval(progressInterval);
});

// Helper function to display circuits on the map and in the list
function displayCircuits(circuits) {
    circuitList.innerHTML = '';
    circuits.forEach((circuit, index) => {
        const listItem = document.createElement('li');
        listItem.innerHTML = `
            <strong>서킷 ${index + 1}</strong>
            <p>길이: ${(circuit.length / 1000).toFixed(2)} km</p>
            <p>점수: ${circuit.score.toFixed(2)}</p>
        `;
        listItem.addEventListener('click', () => {
            const coords = circuit.coordinates.map(c => [c[0], c[1]]);
            const bounds = L.latLngBounds(coords);
            map.fitBounds(bounds, { padding: [50, 50] });
            highlightCircuit(circuit);
        });
        circuitList.appendChild(listItem);
        const polyline = L.polyline(circuit.coordinates, { color: 'gray', weight: 5, opacity: 0.7 }).addTo(map);
        currentCircuitLayers.push(polyline);
    });
}

// Helper function to highlight a specific circuit
function highlightCircuit(selectedCircuit) {
    currentCircuitLayers.forEach(layer => map.removeLayer(layer));
    currentCircuitLayers = [];
    const highlightPolyline = L.polyline(selectedCircuit.coordinates, { color: 'red', weight: 8, opacity: 1 }).addTo(map);
    currentCircuitLayers.push(highlightPolyline);
    L.marker(selectedCircuit.coordinates[0]).addTo(map).bindPopup('피트 레인');
}
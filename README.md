<div align="center">

# F1 Circuit Finder

> 실제 도로망 데이터에서 당신만의 F1 서킷을 찾아보세요.

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

---

## 소개

F1 Circuit Finder는 실제 세계의 도로망 데이터를 분석하여 Formula 1 스타일의 레이스 서킷이 될 수 있는 잠재적인 경로를 찾아내는 프로젝트입니다. 사용자가 지도에서 특정 지역을 선택하면, 해당 지역의 도로 데이터를 기반으로 F1 서킷 규정에 부합하는 후보 경로를 탐색하고 평가하여 제시합니다.

---

## 주요 기능

- **지도 기반 지역 선택**: 사용자는 대화형 지도에서 원하는 지역을 직접 선택하여 서킷 탐색을 시작할 수 있습니다.
- **도로망 필터링**: 4차선 이상의 넓은 도로, 완만한 경사도 등 서킷에 적합한 도로만 필터링하여 분석 대상으로 삼습니다.
- **자동 서킷 탐색**: 그래프 탐색 알고리즘(DFS)을 사용하여 필터링된 도로망 내에서 닫힌 순환 경로(서킷)를 자동으로 찾아냅니다.
- **상세 규정 검증**: 탐색된 경로는 최소/최대 길이(3.2km ~ 7km), 최소 직선 구간 길이(800m 이상), 코너 간 최소 거리 등 F1 서킷의 기술적 요구사항에 따라 검증됩니다.
- **서킷 평가 및 순위**: 검증을 통과한 서킷들은 코너의 수, 고저차, 가장 긴 직선 구간 길이 등을 종합하여 점수를 매기고 순위를 정합니다.
- **결과 시각화**: 최종 후보 서킷들은 지도 위에 경로가 표시되어 시각적으로 확인할 수 있습니다.
- **데이터 캐싱**: 한번 분석한 지역의 도로망 데이터는 캐시 파일로 저장하여, 동일 지역에 대한 재탐색 시 처리 속도를 크게 향상시킵니다.

---

## 기술 스택

#### 백엔드
- **언어**: Python
- **프레임워크**: Flask

#### 지리 데이터 처리
- **라이브러리**: OSMnx, NetworkX
- **외부 API**: Google Maps Elevation API

#### 프론트엔드
- **언어**: HTML, CSS, JavaScript
- **라이브러리**: Leaflet.js

---

## 설치 및 실행 방법

1.  **저장소 복제**
    ```bash
    git clone <repository-url>
    cd F1-circuit-finder
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 변수 설정**
    프로젝트 루트 디렉터리에 `.env` 파일을 생성하고, Google Maps API 키를 다음과 같이 추가합니다.
    ```
    GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
    ```

5.  **애플리케이션 실행**
    ```bash
    python app.py
    ```
    실행 후 웹 브라우저에서 `http://127.0.0.1:5000` 주소로 접속합니다.
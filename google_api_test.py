import os
from dotenv import load_dotenv
import googlemaps

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# API 키를 가져옵니다.
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if not api_key:
    print("❌ 오류: .env 파일에서 GOOGLE_MAPS_API_KEY를 찾을 수 없습니다.")
    print("   프로젝트 루트 디렉토리에 .env 파일이 있고, 키가 올바르게 설정되었는지 확인하세요.")
else:
    print("✅ API 키를 성공적으로 로드했습니다.")
    
    # Google Maps 클라이언트 초기화
    gmaps = googlemaps.Client(key=api_key)

    # 테스트할 샘플 좌표 (서울, 남산타워)
    location = (37.5512, 126.9882)
    
    print(f"\n📡 테스트 좌표 {location}의 고도를 요청합니다...")

    try:
        # 고도 데이터 요청
        elevation_result = gmaps.elevation(location)

        print("\n--- API 응답 결과 ---")
        
        # 결과가 리스트 형태로 반환됩니다.
        if elevation_result and isinstance(elevation_result, list):
            result = elevation_result[0]
            elevation = result.get('elevation')
            resolution = result.get('resolution')
            lat = result.get('location', {}).get('lat')
            lng = result.get('location', {}).get('lng')

            print(f"✔️ 상태: 성공")
            print(f"   - 요청 위치: (lat: {lat:.4f}, lng: {lng:.4f})")
            print(f"   - 고도: {elevation:.2f} 미터")
            print(f"   - 해상도: {resolution:.2f} 미터")
            print("\n🎉 테스트 성공: API 키가 정상적으로 작동합니다.")

        else:
            print("⚠️ 상태: 실패 또는 예기치 않은 응답")
            print("   - 받은 데이터:", elevation_result)

    except googlemaps.exceptions.ApiError as e:
        print("\n--- ❌ API 오류 발생 ---")
        print(f"오류 메시지: {e}")
        print("\nAPI 키가 유효하지 않거나, Google Cloud 프로젝트에서 'Elevation API'가 활성화되지 않았을 수 있습니다.")
        print("Google Cloud 콘솔에서 다음 사항을 확인해 보세요:")
        print("1. API 키가 올바른지")
        print("2. 'Elevation API'가 프로젝트에 사용 설정되었는지")
        print("3. 결제 계정이 프로젝트에 연결되어 있는지")
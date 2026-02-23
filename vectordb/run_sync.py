# vectordb/run_sync.py
import logging
from vectordb.maintenance import run_full_sync

# 로그 설정 (실시간 진행 상황 확인용)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("🌐 LDraw 라이브러리 전체 동기화(ZIP 다운로드 포함)를 시작합니다.")
    print("⚠️ 이 작업은 네트워크 상태와 PC 성능에 따라 수 분이 소요될 수 있습니다.")
    
    success = run_full_sync()
    
    if success:
        print("\n✨ 모든 동기화 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 동기화 중 오류가 발생했습니다. 로그를 확인해 주세요.")

if __name__ == "__main__":
    main()

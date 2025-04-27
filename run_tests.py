# run_tests.py - 프로젝트 인피니티 코인 v2 테스트 실행 스크립트

import os
import sys
import argparse
import unittest
import asyncio
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_run.log"),
        logging.StreamHandler()
    ]
)

# 테스트 파일들
TEST_FILES = {
    'basic': 'test_system.py',
    'integration': 'test_integration.py',
    'full': 'test_full_system.py'
}

def parse_arguments():
    """커맨드 라인 인자 처리"""
    parser = argparse.ArgumentParser(description='프로젝트 인피니티 코인 v2 테스트 실행')
    
    parser.add_argument(
        'test_type',
        nargs='?',
        default='all',
        choices=['all', 'basic', 'integration', 'full'],
        help='실행할 테스트 유형 (기본값: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로깅 활성화'
    )
    
    return parser.parse_args()

def run_unittest_file(test_file):
    """unittest 파일 실행"""
    logging.info(f"{test_file} 테스트 실행 중...")
    
    # 현재 디렉토리를 기준으로 파일 경로 설정
    test_file_path = os.path.join(os.path.dirname(__file__), test_file)
    
    if not os.path.exists(test_file_path):
        logging.error(f"테스트 파일을 찾을 수 없음: {test_file_path}")
        return False
    
    # unittest 실행
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(test_file_path), pattern=os.path.basename(test_file_path))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logging.info(f"{test_file} 테스트 성공")
        return True
    else:
        logging.error(f"{test_file} 테스트 실패")
        return False

def run_asyncio_test_file(test_file):
    """비동기 테스트 파일 실행 (test_full_system.py)"""
    logging.info(f"{test_file} 테스트 실행 중...")
    
    # 현재 디렉토리를 기준으로 파일 경로 설정
    test_file_path = os.path.join(os.path.dirname(__file__), test_file)
    
    if not os.path.exists(test_file_path):
        logging.error(f"테스트 파일을 찾을 수 없음: {test_file_path}")
        return False
    
    # 모듈 임포트 방식으로 실행
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", test_file_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # 비동기 테스트 함수 실행
    try:
        asyncio.run(test_module.run_all_tests())
        logging.info(f"{test_file} 테스트 성공")
        return True
    except Exception as e:
        logging.error(f"{test_file} 테스트 실행 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 테스트 실행
    if args.test_type == 'all':
        logging.info("모든 테스트 실행 중...")
        
        # 기본 컴포넌트 테스트
        basic_result = run_unittest_file(TEST_FILES['basic'])
        
        # 통합 테스트
        integration_result = run_unittest_file(TEST_FILES['integration'])
        
        # 전체 시스템 테스트 (비동기)
        full_result = run_asyncio_test_file(TEST_FILES['full'])
        
        # 결과 요약
        logging.info("\n=== 테스트 결과 요약 ===")
        logging.info(f"기본 컴포넌트 테스트: {'성공' if basic_result else '실패'}")
        logging.info(f"통합 테스트: {'성공' if integration_result else '실패'}")
        logging.info(f"전체 시스템 테스트: {'성공' if full_result else '실패'}")
        
        if basic_result and integration_result and full_result:
            logging.info("✅ 모든 테스트 통과")
            return 0
        else:
            logging.error("❌ 일부 테스트 실패")
            return 1
            
    elif args.test_type == 'basic':
        result = run_unittest_file(TEST_FILES['basic'])
        return 0 if result else 1
        
    elif args.test_type == 'integration':
        result = run_unittest_file(TEST_FILES['integration'])
        return 0 if result else 1
        
    elif args.test_type == 'full':
        result = run_asyncio_test_file(TEST_FILES['full'])
        return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())
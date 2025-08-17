#!/usr/bin/env python3
"""
P&ID 분석기 CLI 사용 예시
Amazon Bedrock 기반 계층적 파이프라인 아키텍처
"""

from pnid_analyzer import PNIDAnalyzer
from PIL import Image
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='P&ID 분석기 CLI')
    parser.add_argument('image_path', help='분석할 P&ID 이미지 경로')
    parser.add_argument('--model', default='claude-3-5-sonnet', 
                       choices=['claude-3-5-sonnet', 'claude-3-5-haiku', 'claude-3-sonnet', 'nova-pro', 'nova-lite'],
                       help='사용할 AI 모델')
    parser.add_argument('--tile-size', type=int, default=512, help='타일 크기 (픽셀)')
    parser.add_argument('--overlap', type=float, default=0.5, help='타일 겹침 비율')
    parser.add_argument('--output', help='결과 저장 경로 (JSON)')
    
    args = parser.parse_args()
    
    try:
        # 분석기 초기화
        print(f"P&ID 분석기 초기화 중... (모델: {args.model})")
        analyzer = PNIDAnalyzer()
        analyzer.set_model(args.model)
        
        # 이미지 로드
        print(f"이미지 로드 중: {args.image_path}")
        image = Image.open(args.image_path)
        print(f"이미지 크기: {image.size}")
        
        # 계층적 분석 실행
        print("계층적 P&ID 분석 시작...")
        result = analyzer.analyze_pnid_hierarchical(
            image, 
            tile_size=args.tile_size, 
            overlap_ratio=args.overlap
        )
        
        # 결과 출력
        print("\n=== 분석 결과 ===")
        print(f"분석 유형: {result['analysis_type']}")
        print(f"총 타일 수: {result['total_tiles']}")
        print(f"처리 시간: {result.get('processing_time', 'N/A')}초")
        
        stats = result.get('statistics', {})
        print(f"탐지된 기호: {stats.get('total_symbols', 0)}개")
        print(f"추출된 텍스트: {stats.get('total_texts', 0)}개")
        print(f"연결 관계: {stats.get('total_connections', 0)}개")
        
        # 주요 기호 출력
        symbols = result.get('symbols', [])
        if symbols:
            print("\n=== 주요 탐지 기호 ===")
            for i, symbol in enumerate(symbols[:5]):  # 상위 5개만
                print(f"{i+1}. {symbol.get('class', 'Unknown')} "
                      f"(태그: {symbol.get('tag_id', 'N/A')}, "
                      f"신뢰도: {symbol.get('confidence', 0):.2f})")
        
        # 결과 저장
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n결과가 {args.output}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

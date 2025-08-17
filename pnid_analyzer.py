import boto3
import json
import base64
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Tuple, Optional
import networkx as nx
import cv2
import time
import random

class PNIDAnalyzer:
    def __init__(self, region_name='us-east-1'):
        self.region_name = region_name
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.models = {
            'claude-3-5-sonnet': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
            'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
            'nova-premier': 'us.amazon.nova-premier-v1:0',  # inference profile
            'nova-pro': 'amazon.nova-pro-v1:0',
            'nova-lite': 'amazon.nova-lite-v1:0'
        }
        self.current_model = 'claude-3-5-sonnet'
    
    def set_region(self, region_name: str):
        """AWS 리전 변경"""
        self.region_name = region_name
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        print(f"[DEBUG] AWS 리전 변경: {region_name}")
        
    def set_model(self, model_name: str):
        if model_name in self.models:
            self.current_model = model_name
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def _encode_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _call_bedrock(self, prompt: str, image_b64: str, model_name: str = None, max_retries: int = 5) -> str:
        model = model_name or self.current_model
        model_id = self.models[model]
        
        print(f"[DEBUG] Using model: {model} ({model_id})")
        print(f"[DEBUG] Prompt: {prompt[:500]}...")
        
        # 디버그 로그를 위한 전역 변수 설정
        if not hasattr(self, 'debug_logs'):
            self.debug_logs = []
        
        if model.startswith('nova'):
            # Nova format (올바른 파라미터 사용)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": {"format": "png", "source": {"bytes": image_b64}}}
                        ]
                    }
                ],
                "inferenceConfig": {"max_new_tokens": 4000, "temperature": 0.1}
            }
        else:
            # Claude format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
                        ]
                    }
                ]
            }
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                result = json.loads(response['body'].read())
                print(f"[DEBUG] Raw response keys: {result.keys()}")
                
                if model.startswith('nova'):
                    response_text = result['output']['message']['content'][0]['text']
                else:
                    response_text = result['content'][0]['text']
                
                print(f"[DEBUG] Response text length: {len(response_text)}")
                print(f"[DEBUG] Response preview: {response_text[:200]}...")
                
                # 디버그 로그 저장
                self.debug_logs.append({
                    'model': model,
                    'model_id': model_id,
                    'prompt': prompt,
                    'response': response_text,
                    'image_b64': image_b64,
                    'timestamp': time.time()
                })
                
                return response_text
                
            except Exception as e:
                error_str = str(e)
                print(f"[ERROR] Attempt {attempt + 1} failed: {error_str}")
                
                # ThrottlingException 또는 rate limit 관련 오류 확인
                if "ThrottlingException" in error_str or "Too many requests" in error_str or "rate" in error_str.lower():
                    if attempt < max_retries - 1:
                        # 지수 백오프: 2^attempt + 랜덤 지터
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"[INFO] Rate limited. Waiting {wait_time:.2f} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[ERROR] Max retries reached. Final error: {error_str}")
                        raise Exception(f"API rate limit exceeded after {max_retries} attempts. Please wait a few minutes and try again.")
                else:
                    # 다른 종류의 오류는 즉시 재발생
                    print(f"[ERROR] Non-throttling error: {error_str}")
                    raise
    
    def identify_interest_regions(self, image: Image.Image) -> List[Dict]:
        """1단계: 저해상도로 관심 영역 식별"""
        # 저해상도로 축소
        low_res = image.copy()
        low_res.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        prompt = """<instructions>
이 P&ID 다이어그램에서 기호(심볼)가 밀집된 "관심 영역"을 식별하세요.
각 영역을 경계 상자 [x1, y1, x2, y2]로 표시하고, 밀도 점수(1-10)를 부여하세요.
</instructions>

<output_format>
{
  "interest_regions": [
    {
      "bbox": [x1, y1, x2, y2],
      "density_score": 8,
      "description": "펌프와 밸브가 밀집된 영역"
    }
  ]
}
</output_format>"""
        
        image_b64 = self._encode_image(low_res)
        response = self._call_bedrock(prompt, image_b64, 'claude-3-5-haiku')
        
        try:
            return json.loads(response)['interest_regions']
        except:
            return []
    
    def analyze_pnid_overview(self, image: Image.Image) -> Dict:
        """전체 P&ID 이미지 개요 분석 (주요 부품 식별)"""
        
        try:
            # 이미지 크기 축소 (빠른 처리를 위해)
            max_size = 1024
            width, height = image.size
            if width > max_size or height > max_size:
                ratio = min(max_size/width, max_size/height)
                new_size = (int(width * ratio), int(height * ratio))
                overview_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                overview_image = image
            
            prompt = """이 P&ID 다이어그램에서 모든 기호(심볼)를 찾아 정확한 위치를 식별하세요.

중요: 기호만 집중해서 찾으세요. 텍스트나 배관은 무시하고 오직 P&ID 기호만 탐지하세요.

P&ID 기호 종류:
- 펌프 (Pump)
- 밸브 (Valve) 
- 탱크/베셀 (Tank/Vessel)
- 열교환기 (Heat Exchanger)
- 계측기 (Instrument)
- 압축기 (Compressor)
- 필터 (Filter)
- 기타 공정 장비

다음 JSON 형식으로 응답하세요:
{
  "symbols": [
    {
      "class": "기호 분류",
      "type": "기호 유형", 
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.8
    }
  ]
}

모든 bbox 좌표는 [x1, y1, x2, y2] 형식으로 정확한 픽셀 좌표를 제공하세요."""
            
            image_b64 = self._encode_image(overview_image)
            response = self._call_bedrock(prompt, image_b64)
            
            print(f"[DEBUG] 개요 분석 응답: {response[:500]}...")
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # 좌표를 원본 이미지 크기로 스케일링
                if overview_image.size != image.size:
                    scale_x = width / overview_image.size[0]
                    scale_y = height / overview_image.size[1]
                    
                    for symbol in result.get('symbols', []):
                        if 'bbox' in symbol and symbol['bbox']:
                            bbox = symbol['bbox']
                            # 1000 스케일 좌표를 개요 이미지 크기로 변환
                            local_bbox = self._convert_1000_scale_coordinates(bbox, overview_image.size[0], overview_image.size[1])
                            # 원본 이미지 크기로 스케일링
                            symbol['bbox'] = [
                                int(local_bbox[0] * scale_x), int(local_bbox[1] * scale_y),
                                int(local_bbox[2] * scale_x), int(local_bbox[3] * scale_y)
                            ]
                            print(f"[DEBUG] 개요 심볼 좌표 변환: {bbox} -> {local_bbox} -> {symbol['bbox']}")
                else:
                    # 이미지 크기가 같으면 1000 스케일을 직접 변환
                    for symbol in result.get('symbols', []):
                        if 'bbox' in symbol and symbol['bbox']:
                            original_bbox = symbol['bbox'].copy()
                            symbol['bbox'] = self._convert_1000_scale_coordinates(symbol['bbox'], width, height)
                            print(f"[DEBUG] 개요 심볼 좌표 변환 (동일 크기): {original_bbox} -> {symbol['bbox']}")
                
                print(f"[DEBUG] 개요 분석 완료 - 주요 장비: {len(result.get('major_equipment', []))}개, 심볼: {len(result.get('symbols', []))}개")
                return result
            else:
                print("[WARNING] 개요 분석 JSON 파싱 실패 - 빈 결과 반환")
                return {
                    "system_overview": "분석 실패",
                    "major_equipment": [],
                    "process_flows": [],
                    "symbols": [],
                    "error": "JSON not found in response", 
                    "raw_response": response[:500]
                }
        except Exception as e:
            print(f"[ERROR] 개요 분석 실패: {e}")
            return {
                "system_overview": "분석 실패",
                "major_equipment": [],
                "process_flows": [],
                "symbols": [],
                "error": str(e), 
                "raw_response": response[:500] if 'response' in locals() else ""
            }
        """단순 그리드 기반 타일링"""
        width, height = image.size
        tiles = []
        tile_id = 0
        
        print(f"[DEBUG] 이미지 크기: {width}x{height}, 타일 크기: {tile_size}, 오버랩: {overlap_pixels}px")
        
        # 스텝 크기 계산 (오버랩 고려)
        step = tile_size - overlap_pixels
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                # 타일 경계 계산
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # 너무 작은 타일은 제외
                if (x2 - x1) >= tile_size // 2 and (y2 - y1) >= tile_size // 2:
                    tiles.append({
                        'id': tile_id,
                        'bbox': [x1, y1, x2, y2],
                        'model': self.current_model
                    })
                    print(f"[DEBUG] 타일 {tile_id}: ({x1},{y1}) → ({x2},{y2}) 크기: {x2-x1}x{y2-y1}")
                    tile_id += 1
                
                x += step
                if x >= width:
                    break
            
            y += step
            if y >= height:
                break
        
        print(f"[DEBUG] 총 {len(tiles)}개 타일 생성")
        return tiles

    
    def generate_grid_tiles(self, image: Image.Image, tile_size: int = 512, overlap_pixels: int = 64) -> List[Dict]:
        """100% 완전 커버리지 타일링 (오버랩 검증 포함)"""
        width, height = image.size
        tiles = []
        tile_id = 0
        
        print(f"[DEBUG] 이미지 크기: {width}x{height}")
        print(f"[DEBUG] 타일 크기: {tile_size}px, 오버랩: {overlap_pixels}px")
        
        # 스텝 크기 계산 (오버랩 적용)
        step = tile_size - overlap_pixels
        print(f"[DEBUG] 실제 스텝 크기: {step}px (타일 간격)")
        
        if step <= 0:
            print(f"[ERROR] 잘못된 설정: 오버랩({overlap_pixels})이 타일 크기({tile_size})보다 크거나 같음")
            step = tile_size // 2  # 안전한 기본값
        
        # 100% 커버리지를 위한 타일 개수 계산
        cols = max(1, int(np.ceil(width / step)))
        rows = max(1, int(np.ceil(height / step)))
        
        print(f"[DEBUG] 타일 배치: {cols}열 x {rows}행")
        print(f"[DEBUG] 예상 오버랩 영역: 가로 {(cols-1)*overlap_pixels}px, 세로 {(rows-1)*overlap_pixels}px")
        
        # 오버랩 검증용 카운터
        overlap_map = np.zeros((height, width), dtype=int)
        
        for row in range(rows):
            for col in range(cols):
                # 타일 시작 좌표 (오버랩 고려)
                x1 = col * step
                y1 = row * step
                
                # 타일 끝 좌표
                x2 = min(x1 + tile_size, width)
                y2 = min(y1 + tile_size, height)
                
                # 마지막 행/열의 경우 이미지 끝까지 확장
                if col == cols - 1:
                    x2 = width
                if row == rows - 1:
                    y2 = height
                
                # 실제 타일 크기
                actual_width = x2 - x1
                actual_height = y2 - y1
                
                # 오버랩 계산
                overlap_left = overlap_pixels if col > 0 else 0
                overlap_top = overlap_pixels if row > 0 else 0
                overlap_right = overlap_pixels if col < cols - 1 else 0
                overlap_bottom = overlap_pixels if row < rows - 1 else 0
                
                tiles.append({
                    'id': tile_id,
                    'bbox': [x1, y1, x2, y2],
                    'model': self.current_model,
                    'row': row,
                    'col': col,
                    'step_x': col * step,
                    'step_y': row * step,
                    'actual_size': f"{actual_width}x{actual_height}",
                    'overlaps': {
                        'left': overlap_left,
                        'top': overlap_top,
                        'right': overlap_right,
                        'bottom': overlap_bottom
                    }
                })
                
                # 오버랩 맵 업데이트
                overlap_map[y1:y2, x1:x2] += 1
                
                overlap_info = f"L{overlap_left}T{overlap_top}R{overlap_right}B{overlap_bottom}"
                print(f"[DEBUG] 타일 {tile_id} ({row},{col}): ({x1},{y1})→({x2},{y2}) 크기:{actual_width}x{actual_height} 오버랩:{overlap_info}")
                tile_id += 1
        
        # 오버랩 통계
        unique_counts = np.unique(overlap_map, return_counts=True)
        print(f"[DEBUG] 오버랩 통계:")
        for count, pixels in zip(unique_counts[0], unique_counts[1]):
            if count == 0:
                print(f"  - 커버되지 않음: {pixels}픽셀")
            elif count == 1:
                print(f"  - 1번 커버: {pixels}픽셀")
            else:
                print(f"  - {count}번 오버랩: {pixels}픽셀")
        
        # 100% 커버리지 검증
        uncovered_pixels = np.sum(overlap_map == 0)
        if uncovered_pixels > 0:
            print(f"[ERROR] {uncovered_pixels}픽셀이 커버되지 않음!")
        else:
            print(f"[SUCCESS] 100% 완전 커버리지 달성!")
        
        print(f"[DEBUG] 총 {len(tiles)}개 타일 생성 완료")
        return tiles
    
    def _find_uncovered_regions(self, coverage_map, width, height):
        """누락된 영역 찾기"""
        uncovered_regions = []
        
        # 간단한 구현: 누락된 픽셀이 있으면 전체를 하나의 타일로 처리
        uncovered_y, uncovered_x = np.where(~coverage_map)
        
        if len(uncovered_y) > 0:
            min_x, max_x = np.min(uncovered_x), np.max(uncovered_x)
            min_y, max_y = np.min(uncovered_y), np.max(uncovered_y)
            uncovered_regions.append([min_x, min_y, max_x + 1, max_y + 1])
        
        return uncovered_regions
    
    def _detect_coordinate_scale(self, bbox, image_width, image_height):
        """좌표 스케일 자동 감지 (1000 스케일 vs 실제 픽셀)"""
        if not bbox or len(bbox) != 4:
            return "unknown", bbox
        
        x1, y1, x2, y2 = bbox
        
        # 1000 스케일인지 확인 (좌표가 모두 1000 이하)
        if max(x1, y1, x2, y2) <= 1000:
            return "1000_scale", bbox
        
        # 실제 픽셀 좌표인지 확인 (이미지 크기 범위 내)
        if x2 <= image_width and y2 <= image_height:
            return "pixel_scale", bbox
        
        # 애매한 경우 1000 스케일로 가정
        return "1000_scale", bbox
    
    def _convert_1000_scale_coordinates(self, bbox, image_width, image_height):
        """AI 모델의 1000 스케일 좌표를 실제 이미지 좌표로 변환"""
        if not bbox or len(bbox) != 4:
            return bbox
        
        # 1000 스케일을 실제 이미지 크기로 변환
        x1 = int((bbox[0] / 1000.0) * image_width)
        y1 = int((bbox[1] / 1000.0) * image_height)
        x2 = int((bbox[2] / 1000.0) * image_width)
        y2 = int((bbox[3] / 1000.0) * image_height)
        
        return [x1, y1, x2, y2]
    
    def _convert_to_global_coordinates(self, tile_result: Dict, offset_x: int, offset_y: int, tile_width: int, tile_height: int):
        """타일 내 1000 스케일 좌표를 전역 좌표로 변환"""
        
        # 심볼 좌표 변환: 1000 스케일 → 타일 좌표 → 전역 좌표
        for symbol in tile_result.get('symbols', []):
            if 'bbox' in symbol and symbol['bbox'] and len(symbol['bbox']) == 4:
                bbox = symbol['bbox']
                # 1단계: 1000 스케일 → 타일 좌표
                local_bbox = self._convert_1000_scale_coordinates(bbox, tile_width, tile_height)
                # 2단계: 타일 좌표 → 전역 좌표
                symbol['bbox'] = [
                    local_bbox[0] + offset_x,
                    local_bbox[1] + offset_y,
                    local_bbox[2] + offset_x,
                    local_bbox[3] + offset_y
                ]
                print(f"[DEBUG] 심볼 좌표 변환: {bbox} → {local_bbox} → {symbol['bbox']}")
        
        # 텍스트 좌표 변환
        for text in tile_result.get('texts', []):
            if 'bbox' in text and text['bbox'] and len(text['bbox']) == 4:
                bbox = text['bbox']
                local_bbox = self._convert_1000_scale_coordinates(bbox, tile_width, tile_height)
                text['bbox'] = [
                    local_bbox[0] + offset_x,
                    local_bbox[1] + offset_y,
                    local_bbox[2] + offset_x,
                    local_bbox[3] + offset_y
                ]
        
        # 라인 좌표 변환
        for line in tile_result.get('lines', []):
            # path 좌표 변환
            if 'path' in line and line['path']:
                new_path = []
                for point in line['path']:
                    if len(point) == 2:
                        if use_1000_scale:
                            # 1000 스케일 변환
                            local_x = int((point[0] / 1000.0) * tile_width)
                            local_y = int((point[1] / 1000.0) * tile_height)
                            new_path.append([local_x + offset_x, local_y + offset_y])
                        else:
                            new_path.append([point[0] + offset_x, point[1] + offset_y])
                    else:
                        new_path.append(point)
                line['path'] = new_path
            
            # bbox 좌표 변환
            if 'bbox' in line and line['bbox']:
                bbox = line['bbox']
                if len(bbox) == 4:
                    if use_1000_scale:
                        local_bbox = self._convert_1000_scale_coordinates(bbox, tile_width, tile_height)
                        line['bbox'] = [
                            local_bbox[0] + offset_x,
                            local_bbox[1] + offset_y,
                            local_bbox[2] + offset_x,
                            local_bbox[3] + offset_y
                        ]
                    else:
                        line['bbox'] = [
                            bbox[0] + offset_x,
                            bbox[1] + offset_y,
                            bbox[2] + offset_x,
                            bbox[3] + offset_y
                        ]
        """단순 그리드 기반 타일링"""
        width, height = image.size
        tiles = []
        tile_id = 0
        
        print(f"[DEBUG] 이미지 크기: {width}x{height}, 타일 크기: {tile_size}, 오버랩: {overlap_pixels}px")
        
        # 스텝 크기 계산 (오버랩 고려)
        step = tile_size - overlap_pixels
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                # 타일 경계 계산
                x1 = x
                y1 = y
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                # 너무 작은 타일은 제외
                if (x2 - x1) >= tile_size // 2 and (y2 - y1) >= tile_size // 2:
                    tiles.append({
                        'id': tile_id,
                        'bbox': [x1, y1, x2, y2],
                        'model': self.current_model
                    })
                    print(f"[DEBUG] 타일 {tile_id}: ({x1},{y1}) → ({x2},{y2}) 크기: {x2-x1}x{y2-y1}")
                    tile_id += 1
                
                x += step
                if x >= width:
                    break
            
            y += step
            if y >= height:
                break
        
        print(f"[DEBUG] 총 {len(tiles)}개 타일 생성")
        return tiles
    
    def integrate_overview_and_details(self, overview: Dict, details: Dict) -> Dict:
        """개요 분석과 상세 분석 결과 통합 (중복 제거 없음)"""
        
        # 개요에서 심볼들 추출
        overview_symbols = overview.get('symbols', [])
        
        # 상세 분석 결과
        detail_symbols = details.get('symbols', [])
        detail_texts = details.get('texts', [])
        detail_lines = details.get('lines', [])
        
        print(f"[DEBUG] 통합 전 - 개요 심볼: {len(overview_symbols)}개")
        print(f"[DEBUG] 통합 전 - 상세 심볼: {len(detail_symbols)}개")
        print(f"[DEBUG] 통합 전 - 상세 텍스트: {len(detail_texts)}개")
        print(f"[DEBUG] 통합 전 - 상세 라인: {len(detail_lines)}개")
        
        # 개요 분석이 실패했거나 심볼이 없으면 상세 분석 결과만 사용
        if not overview_symbols or len(overview_symbols) == 0:
            print("[DEBUG] 개요 분석 결과가 없음 - 상세 분석 결과만 사용")
            
            # 상세 심볼에 출처 표시
            for symbol in detail_symbols:
                if 'source' not in symbol:
                    symbol['source'] = 'detail'
            
            print(f"[DEBUG] 개요 없음 - 최종 심볼: {len(detail_symbols)}개")
            
            return {
                'symbols': detail_symbols,
                'texts': detail_texts,
                'lines': detail_lines,
                'overview_info': {
                    'system_overview': overview.get('system_overview', ''),
                    'major_equipment': overview.get('major_equipment', []),
                    'process_flows': overview.get('process_flows', [])
                }
            }
        
        # 모든 심볼을 단순 병합 (중복 제거 없음)
        all_symbols = []
        
        # 개요 심볼 추가 (출처 표시)
        for symbol in overview_symbols:
            symbol_copy = symbol.copy()
            symbol_copy['source'] = 'overview'
            all_symbols.append(symbol_copy)
        
        # 상세 심볼 추가 (출처 표시)
        for symbol in detail_symbols:
            symbol_copy = symbol.copy()
            symbol_copy['source'] = 'detail'
            all_symbols.append(symbol_copy)
        
        print(f"[DEBUG] 통합 후 - 최종 심볼: {len(all_symbols)}개 (개요 {len(overview_symbols)} + 상세 {len(detail_symbols)})")
        print(f"[DEBUG] 통합 후 - 최종 텍스트: {len(detail_texts)}개")
        print(f"[DEBUG] 통합 후 - 최종 라인: {len(detail_lines)}개")
        
        return {
            'symbols': all_symbols,
            'texts': detail_texts,
            'lines': detail_lines,
            'overview_info': {
                'system_overview': overview.get('system_overview', ''),
                'major_equipment': overview.get('major_equipment', []),
                'process_flows': overview.get('process_flows', [])
            }
        }
        """개별 타일 분석"""
        x1, y1, x2, y2 = tile_info['bbox']
        tile_image = image.crop((x1, y1, x2, y2))
        
        prompt = f"""이 P&ID 타일에서 모든 기호(심볼)를 찾아 정확한 위치를 식별하세요.

중요: 기호만 집중해서 찾으세요. 텍스트나 배관은 무시하고 오직 P&ID 기호만 탐지하세요.

P&ID 기호 종류:
- 펌프 (Pump)
- 밸브 (Valve) 
- 탱크/베셀 (Tank/Vessel)
- 열교환기 (Heat Exchanger)
- 계측기 (Instrument)
- 압축기 (Compressor)
- 필터 (Filter)
- 기타 공정 장비

반드시 다음 JSON 형식으로 응답하세요:

{{
  "tile_id": {tile_info['id']},
  "symbols": [
    {{
      "id": "symbol_1",
      "class": "Pump",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }}
  ]
}}"""
        
        image_b64 = self._encode_image(tile_image)
        response = self._call_bedrock(prompt, image_b64, tile_info['model'])
        
        print(f"[DEBUG] Tile {tile_info['id']} response: {response[:300]}...")
        
        try:
            # JSON 추출 시도
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                print(f"[DEBUG] Parsed JSON successfully for tile {tile_info['id']}")
            else:
                print(f"[DEBUG] No JSON found in response for tile {tile_info['id']}")
                result = {"tile_id": tile_info['id'], "symbols": [], "texts": [], "lines": []}
            
            # 좌표를 전역 좌표계로 변환
            self._convert_to_global_coordinates(result, x1, y1)
            return result
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parsing failed for tile {tile_info['id']}: {str(e)}")
            print(f"[ERROR] Response was: {response}")
            
            # 간단한 패턴 매칭으로 기호 추출 시도
            symbols = []
            if any(word in response.lower() for word in ['pump', 'valve', 'tank', 'vessel']):
                symbols.append({
                    "id": f"symbol_{tile_info['id']}_1",
                    "class": "Equipment",
                    "bbox": [x1+10, y1+10, x1+50, y1+50],
                    "confidence": 0.5,
                    "tag_id": "Unknown"
                })
            
            return {
                "tile_id": tile_info.get('id', 'unknown'), 
                "symbols": symbols, 
                "texts": [], 
                "lines": [], 
                "error": str(e),
                "raw_response": response[:500] if 'response' in locals() else ""
            }
    
    def analyze_tile(self, image: Image.Image, tile_info: Dict) -> Dict:
        """개선된 개별 타일 분석"""
        x1, y1, x2, y2 = tile_info['bbox']
        tile_image = image.crop((x1, y1, x2, y2))
        tile_width, tile_height = tile_image.size
        
        # 개선된 프롬프트 - 더 구체적이고 상세함
        prompt = f"""이 P&ID 도면 타일을 매우 세심하게 분석하여 모든 기호(심볼)를 찾아주세요.

이미지 크기: {tile_width} x {tile_height} 픽셀

찾아야 할 P&ID 기호들:
1. 펌프 (원심펌프, 기어펌프, 피스톤펌프 등)
2. 밸브 (게이트밸브, 볼밸브, 체크밸브, 제어밸브 등)
3. 탱크/베셀 (저장탱크, 반응기, 분리기 등)
4. 열교환기 (쉘앤튜브, 플레이트 등)
5. 계측기 (압력계, 온도계, 유량계, 레벨계 등)
6. 압축기/팬/블로워
7. 필터/스트레이너
8. 배관 피팅 (엘보, 티, 리듀서 등)
9. 기타 공정 장비

중요 지침:
- 아주 작은 기호도 놓치지 마세요
- 부분적으로 보이는 기호도 포함하세요
- 각 기호의 정확한 경계를 찾으세요
- 좌표는 0~1000 스케일로 제공하세요 (실제 픽셀이 아님)

JSON 형식으로 응답:
{{
  "symbols": [
    {{
      "id": "symbol_1",
      "class": "구체적 기호명 (예: Centrifugal Pump, Gate Valve)",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "tag_id": "태그번호 (있다면)"
    }}
  ]
}}

bbox 좌표는 반드시 [x1, y1, x2, y2] 형식이며, 0~1000 스케일입니다."""
        
        try:
            image_b64 = self._encode_image(tile_image)
            response = self._call_bedrock(prompt, image_b64, tile_info.get('model', self.current_model))
            
            tile_id = tile_info.get('id', 'unknown')
            print(f"[DEBUG] 타일 {tile_id} 응답 길이: {len(response)} 문자")
            
            # 더 강력한 JSON 추출
            result = self._extract_json_from_response(response)
            
            if result and 'symbols' in result:
                symbols = result['symbols']
                print(f"[DEBUG] 타일 {tile_id} 파싱된 심볼: {len(symbols)}개")
                
                # 1000 스케일 좌표를 실제 좌표로 변환
                for symbol in symbols:
                    if 'bbox' in symbol:
                        bbox = symbol['bbox']
                        # 1000 스케일 → 타일 좌표
                        symbol['bbox'] = self._convert_1000_scale_coordinates(bbox, tile_width, tile_height)
                        # 타일 좌표 → 전역 좌표
                        symbol['bbox'] = [
                            symbol['bbox'][0] + x1,
                            symbol['bbox'][1] + y1,
                            symbol['bbox'][2] + x1,
                            symbol['bbox'][3] + y1
                        ]
                        symbol['source'] = f'tile_{tile_id}'
                        symbol['source_tile'] = tile_id
                
                print(f"[DEBUG] 타일 {tile_id} 좌표 변환 완료")
                if symbols:
                    print(f"[DEBUG] 첫 심볼 최종 좌표: {symbols[0].get('bbox', 'No bbox')}")
                
                return {
                    "tile_id": tile_id,
                    "symbols": symbols,
                    "texts": result.get('texts', []),
                    "lines": result.get('lines', [])
                }
            else:
                print(f"[WARNING] 타일 {tile_id} 유효한 결과 없음")
                return {"tile_id": tile_id, "symbols": [], "texts": [], "lines": []}
            
        except Exception as e:
            tile_id = tile_info.get('id', 'unknown')
            print(f"[ERROR] 타일 {tile_id} 분석 실패: {e}")
            return {"tile_id": tile_id, "symbols": [], "texts": [], "lines": [], "error": str(e)}
        except Exception as e:
            tile_id = tile_info.get('id', 'unknown')
            print(f"[ERROR] 타일 {tile_id} 분석 실패: {e}")
            return {
                "tile_id": tile_id,
                "symbols": [], 
                "texts": [], 
                "lines": [], 
                "error": str(e),
                "raw_response": response[:500] if 'response' in locals() else ""
            }
    
    def merge_overlapping_detections(self, all_results: List[Dict], iou_threshold: float = 0.5) -> Dict:
        """중복 제거 없이 모든 결과 병합"""
        all_symbols = []
        all_texts = []
        all_lines = []
        
        print(f"[DEBUG] 병합 시작 - 입력 결과 수: {len(all_results)}개 (중복 제거 비활성화)")
        
        for i, result in enumerate(all_results):
            symbols = result.get('symbols', [])
            texts = result.get('texts', [])
            lines = result.get('lines', [])
            
            print(f"[DEBUG] 결과 {i}: 심볼 {len(symbols)}개, 텍스트 {len(texts)}개, 라인 {len(lines)}개")
            
            # 타일 ID 정보 추가하여 출처 추적
            for symbol in symbols:
                symbol['source_tile'] = i
            for text in texts:
                text['source_tile'] = i
            for line in lines:
                line['source_tile'] = i
            
            all_symbols.extend(symbols)
            all_texts.extend(texts)
            all_lines.extend(lines)
        
        print(f"[DEBUG] 병합 완료 - 총 심볼: {len(all_symbols)}개, 텍스트: {len(all_texts)}개, 라인: {len(all_lines)}개")
        print(f"[DEBUG] 중복 제거 건너뜀 - 모든 객체 보존")
        
        return {
            'symbols': all_symbols,
            'texts': all_texts,
            'lines': all_lines
        }
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """비최대 억제 알고리즘 적용"""
        if not detections:
            return []
        
        # bbox가 없는 항목들은 별도 처리
        valid_detections = []
        no_bbox_detections = []
        
        for det in detections:
            if 'bbox' in det and det['bbox'] and len(det['bbox']) == 4:
                valid_detections.append(det)
            else:
                no_bbox_detections.append(det)
        
        print(f"[DEBUG] NMS - bbox 있음: {len(valid_detections)}개, bbox 없음: {len(no_bbox_detections)}개")
        
        # bbox 좌표 범위 확인 (디버그용)
        if valid_detections:
            x_coords = [bbox[0] for det in valid_detections for bbox in [det['bbox']]]
            y_coords = [bbox[1] for det in valid_detections for bbox in [det['bbox']]]
            print(f"[DEBUG] 좌표 범위 - X: {min(x_coords)}~{max(x_coords)}, Y: {min(y_coords)}~{max(y_coords)}")
        
        if not valid_detections:
            return no_bbox_detections
        
        # 신뢰도 순으로 정렬
        sorted_detections = sorted(valid_detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        removed_count = 0
        
        while sorted_detections:
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # 현재 탐지와 겹치는 것들 제거
            remaining = []
            for det in sorted_detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
                else:
                    removed_count += 1
                    print(f"[DEBUG] 중복 제거: {det.get('class', 'Unknown')} (IoU: {iou:.3f}) bbox1={current['bbox']} bbox2={det['bbox']}")
            sorted_detections = remaining
        
        print(f"[DEBUG] NMS 완료 - 유지: {len(keep)}개, 제거: {removed_count}개")
        
        # bbox 없는 항목들도 포함
        return keep + no_bbox_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """IoU (Intersection over Union) 계산"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_high_interest_regions(self, image: Image.Image) -> List[Dict]:
        """심볼 밀도가 높은 관심 영역 검출"""
        width, height = image.size
        
        # 이미지를 그리드로 나누어 심볼 밀도 추정
        grid_size = 8
        cell_w, cell_h = width // grid_size, height // grid_size
        roi_regions = []
        
        # OpenCV로 변환하여 특징점 검출
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 각 그리드 셀의 특징점 밀도 계산
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * cell_w, i * cell_h
                x2, y2 = min((j + 1) * cell_w, width), min((i + 1) * cell_h, height)
                
                cell = gray[y1:y2, x1:x2]
                
                # 특징점 검출 (코너, 엣지)
                corners = cv2.goodFeaturesToTrack(cell, maxCorners=100, qualityLevel=0.01, minDistance=10)
                edges = cv2.Canny(cell, 50, 150)
                
                # 밀도 계산
                corner_density = len(corners) if corners is not None else 0
                edge_density = np.sum(edges > 0) / (cell.shape[0] * cell.shape[1])
                
                # 고밀도 영역 판정
                if corner_density > 20 or edge_density > 0.1:
                    roi_regions.append({
                        'bbox': [x1, y1, x2, y2],
                        'density': float(corner_density + edge_density * 1000),  # numpy float64 → float 변환
                        'type': 'high_interest'
                    })
        
        print(f"[DEBUG] ROI 검출 완료: {len(roi_regions)}개 관심 영역")
        return roi_regions

    def generate_multiscale_adaptive_tiles(self, image: Image.Image, roi_regions: List[Dict]) -> List[Dict]:
        """멀티스케일 적응형 타일 생성"""
        width, height = image.size
        all_tiles = []
        
        # 기본 타일 크기들 (작은 심볼 → 큰 심볼)
        tile_sizes = [256, 384, 512]
        
        for tile_size in tile_sizes:
            # ROI 영역에 대한 적응형 오버랩
            for roi in roi_regions:
                x1, y1, x2, y2 = roi['bbox']
                density = roi['density']
                
                # 밀도에 따른 오버랩 조정
                if density > 50:  # 고밀도
                    overlap = tile_size // 2  # 50% 오버랩
                elif density > 20:  # 중밀도
                    overlap = tile_size // 3  # 33% 오버랩
                else:  # 저밀도
                    overlap = tile_size // 6  # 17% 오버랩
                
                # ROI 영역 내 타일 생성
                step = tile_size - overlap
                for y in range(y1, y2, step):
                    for x in range(x1, x2, step):
                        tile_x2 = min(x + tile_size, width)
                        tile_y2 = min(y + tile_size, height)
                        
                        if tile_x2 - x >= tile_size // 2 and tile_y2 - y >= tile_size // 2:
                            all_tiles.append({
                                'bbox': [x, y, tile_x2, tile_y2],
                                'size': tile_size,
                                'overlap': overlap,
                                'roi_density': density
                            })
        
        # 전체 이미지 커버리지를 위한 기본 그리드 추가
        base_tiles = self.generate_grid_tiles(image, 384, 128)
        for tile in base_tiles:
            all_tiles.append({
                'bbox': tile,
                'size': 384,
                'overlap': 128,
                'roi_density': 0
            })
        
        print(f"[DEBUG] 멀티스케일 타일 생성 완료: {len(all_tiles)}개")
        return all_tiles
    
    def build_knowledge_graph(self, merged_data: Dict) -> nx.DiGraph:
        """지식 그래프 구축"""
        G = nx.DiGraph()
        
        # 노드 추가 (기호들)
        for symbol in merged_data['symbols']:
            G.add_node(symbol['id'], 
                      symbol_class=symbol['class'],
                      bbox=symbol['bbox'],
                      tag_id=symbol.get('tag_id', ''),
                      confidence=symbol.get('confidence', 0))
        
        # 연결 관계 추론 (라인 기반)
        for line in merged_data['lines']:
            connections = line.get('connections', [])
            if len(connections) >= 2:
                for i in range(len(connections) - 1):
                    G.add_edge(connections[i], connections[i+1], 
                             line_type=line['type'],
                             line_id=line['id'])
        
        return G
    
    def analyze_pnid_hierarchical(self, image: Image.Image, tile_size: int = 384, overlap_pixels: int = 128, iou_threshold: float = 0.2) -> Dict:
        """계층적 P&ID 분석 (2단계: 전체 → 타일링)"""
        
        try:
            # 1단계: 전체 이미지 개요 분석
            print("[DEBUG] 1단계: 전체 이미지 개요 분석 시작")
            overview_result = self.analyze_pnid_overview(image)
            
            # 2단계: ROI 기반 관심 영역 검출
            print("[DEBUG] 2단계: ROI 기반 관심 영역 검출")
            roi_regions = self.detect_high_interest_regions(image)
            
            # 3단계: 멀티스케일 적응형 타일링
            print("[DEBUG] 3단계: 멀티스케일 적응형 타일링")
            tiles = self.generate_multiscale_adaptive_tiles(image, roi_regions)
            
            # 4단계: 병렬 타일 분석
            all_results = []
            for i, tile_info in enumerate(tiles):
                try:
                    print(f"[DEBUG] 타일 {i} 분석 시작: {tile_info['bbox']} (크기: {tile_info['size']}, 밀도: {tile_info['roi_density']:.1f})")
                    result = self.analyze_tile(image, tile_info)  # tile_info 전체 전달
                    
                    # 타일 결과 상세 확인
                    symbols_count = len(result.get('symbols', []))
                    texts_count = len(result.get('texts', []))
                    lines_count = len(result.get('lines', []))
                    
                    print(f"[DEBUG] 타일 {i} 결과: 심볼 {symbols_count}개, 텍스트 {texts_count}개, 라인 {lines_count}개")
                    
                    # 심볼이 있으면 첫 번째 심볼 정보 출력
                    if symbols_count > 0:
                        first_symbol = result['symbols'][0]
                        print(f"[DEBUG] 타일 {i} 첫 번째 심볼: {first_symbol.get('class', 'Unknown')} - {first_symbol.get('bbox', 'No bbox')}")
                    
                    all_results.append(result)
                except Exception as e:
                    print(f"[ERROR] 타일 {i} 분석 실패: {e}")
                    print(f"[ERROR] 타일 정보: {tile_info}")
                    # 빈 결과로 계속 진행
                    all_results.append({"tile_id": i, "symbols": [], "texts": [], "lines": [], "error": str(e)})
            
            # 4단계: 결과 병합
            print(f"[DEBUG] 4단계: 결과 병합 시작")
            merged_data = self.merge_overlapping_detections(all_results, iou_threshold)
            print(f"[DEBUG] 4단계 완료 - 병합된 심볼: {len(merged_data.get('symbols', []))}개")
            
            # 5단계: 개요와 상세 결과 통합
            print(f"[DEBUG] 5단계: 개요-상세 통합 시작")
            integrated_data = self.integrate_overview_and_details(overview_result, merged_data)
            print(f"[DEBUG] 5단계 완료 - 통합된 심볼: {len(integrated_data.get('symbols', []))}개")
            
            # 6단계: 지식 그래프 구축 (비활성화)
            # knowledge_graph = self.build_knowledge_graph(integrated_data)
            knowledge_graph = None
            
            # 최종 결과 검증
            final_symbols = integrated_data.get('symbols', [])
            final_texts = integrated_data.get('texts', [])
            final_lines = integrated_data.get('lines', [])
            
            print(f"[DEBUG] 최종 결과 검증:")
            print(f"  - 최종 심볼: {len(final_symbols)}개")
            print(f"  - 최종 텍스트: {len(final_texts)}개") 
            print(f"  - 최종 라인: {len(final_lines)}개")
            
            # 심볼이 있으면 첫 몇 개 샘플 출력
            if final_symbols:
                print(f"[DEBUG] 심볼 샘플 (처음 3개):")
                for i, symbol in enumerate(final_symbols[:3]):
                    print(f"  {i+1}. {symbol.get('class', 'Unknown')} - {symbol.get('tag_id', 'No ID')} - {symbol.get('source', 'No source')}")
            else:
                print(f"[WARNING] 최종 심볼이 0개입니다!")
            
            return {
                'analysis_type': '계층적 P&ID 분석 (2단계)',
                'overview': overview_result,
                'symbols': final_symbols,
                'texts': final_texts, 
                'lines': final_lines,
                'knowledge_graph': knowledge_graph,
                'statistics': {
                    'total_tiles': len(tiles),
                    'overview_symbols': len(overview_result.get('symbols', [])),
                    'detail_symbols': len(merged_data.get('symbols', [])),
                    'final_symbols': len(final_symbols),
                    'total_texts': len(final_texts),
                    'total_lines': len(final_lines)
                }
            }
        except Exception as e:
            print(f"[ERROR] 계층적 분석 전체 실패: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """AI 응답에서 JSON을 강력하게 추출"""
        try:
            # 방법 1: 표준 JSON 추출
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # 방법 2: 코드 블록 내 JSON 찾기
            import re
            code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(code_block_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 방법 3: symbols 배열만 찾기
            symbols_pattern = r'"symbols"\s*:\s*\[(.*?)\]'
            match = re.search(symbols_pattern, response, re.DOTALL)
            if match:
                symbols_str = f'{{"symbols": [{match.group(1)}]}}'
                return json.loads(symbols_str)
            
            print(f"[WARNING] JSON 추출 실패, 응답: {response[:300]}...")
            return {"symbols": []}
            
        except Exception as e:
            print(f"[ERROR] JSON 추출 중 오류: {e}")
            return {"symbols": []}
        # 1단계: 적응형 타일링
        tiles = self.generate_adaptive_tiles(image, tile_size, overlap_ratio)
        
        # 2단계: 병렬 타일 분석
        all_results = []
        for tile in tiles:
            result = self.analyze_tile(image, tile)
            all_results.append(result)
        
        # 3단계: 결과 병합
        merged_data = self.merge_overlapping_detections(all_results, iou_threshold)
        
        # 4단계: 지식 그래프 구축
        knowledge_graph = self.build_knowledge_graph(merged_data)
        
        return {
            'analysis_type': '계층적 P&ID 분석',
            'total_tiles': len(tiles),
            'symbols': merged_data['symbols'],
            'texts': merged_data['texts'],
            'lines': merged_data['lines'],
            'knowledge_graph': {
                'nodes': list(knowledge_graph.nodes(data=True)),
                'edges': list(knowledge_graph.edges(data=True))
            },
            'statistics': {
                'total_symbols': len(merged_data['symbols']),
                'total_texts': len(merged_data['texts']),
                'total_connections': knowledge_graph.number_of_edges()
            }
        }

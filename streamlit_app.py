import streamlit as st
import json
import os
import io
import base64
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pnid_analyzer import PNIDAnalyzer
import time
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="P&ID 분석기 - Amazon Bedrock 기반",
    page_icon="🔧",
    layout="wide"
)

# 세션 상태 초기화
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = PNIDAnalyzer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'saved_results' not in st.session_state:
    st.session_state.saved_results = []
    st.session_state.saved_results = []

def load_saved_results():
    """저장된 결과 불러오기"""
    saved_dir = "saved_results"
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        return []
    
    results = []
    for filename in os.listdir(saved_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(saved_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        'filename': filename,
                        'timestamp': data.get('timestamp', ''),
                        'model': data.get('model', ''),
                        'total_symbols': data.get('statistics', {}).get('total_symbols', 0)
                    })
            except:
                continue
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

def save_analysis_result(result, image, model_name):
    """분석 결과 저장"""
    saved_dir = "saved_results"
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pnid_analysis_{timestamp}.json"
    
    # 이미지를 base64로 인코딩
    import base64
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    save_data = {
        'timestamp': timestamp,
        'model': model_name,
        'image_data': image_b64,
        'analysis_result': result
    }
    
    with open(os.path.join(saved_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    return filename

def visualize_knowledge_graph(kg_data):
    """지식 그래프 시각화"""
    if not kg_data or not kg_data.get('nodes'):
        st.warning("지식 그래프 데이터가 없습니다.")
        return
    
    # NetworkX 그래프 생성
    G = nx.DiGraph()
    
    # 노드 추가
    for node_id, node_data in kg_data['nodes']:
        G.add_node(node_id, **node_data)
    
    # 엣지 추가
    for source, target, edge_data in kg_data['edges']:
        G.add_edge(source, target, **edge_data)
    
    # 레이아웃 계산
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Plotly 그래프 생성
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=2, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # 노드 정보
        node_data = G.nodes[node]
        symbol_class = node_data.get('symbol_class', 'Unknown')
        tag_id = node_data.get('tag_id', node)
        node_text.append(f"{tag_id}<br>{symbol_class}")
        
        # 색상 매핑
        if 'Pump' in symbol_class:
            node_color.append('red')
        elif 'Valve' in symbol_class:
            node_color.append('blue')
        elif 'Controller' in symbol_class:
            node_color.append('green')
        else:
            node_color.append('gray')
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=node_text,
                           textposition="middle center",
                           marker=dict(size=20,
                                     color=node_color,
                                     line=dict(width=2, color='black')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='P&ID 지식 그래프',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002 ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🔧 P&ID 분석기 - Amazon Bedrock 기반")
    st.markdown("**계층적 파이프라인 아키텍처를 통한 고성능 P&ID 도면 분석**")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["🆕 새 분석", "📁 저장된 결과", "⚙️ 설정"])
    
    with tab1:
        st.header("새로운 P&ID 분석")
        
        # 모델 선택
        # 모델 및 리전 선택
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            model_options = {
                'Claude 3.5 Sonnet': 'claude-3-5-sonnet',
                'Claude 3 Sonnet': 'claude-3-sonnet',
                'Claude 3 Haiku': 'claude-3-haiku',
                'Amazon Nova Premier': 'nova-premier',
                'Amazon Nova Pro': 'nova-pro',
                'Amazon Nova Lite': 'nova-lite'
            }
            selected_model = st.selectbox(
                "AI 모델 선택",
                options=list(model_options.keys()),
                help="분석에 사용할 AI 모델을 선택하세요"
            )
            st.session_state.analyzer.set_model(model_options[selected_model])
        
        with col2:
            region_options = {
                'US East (N. Virginia)': 'us-east-1',
                'US West (Oregon)': 'us-west-2',
                'Europe (Frankfurt)': 'eu-central-1',
                'Asia Pacific (Tokyo)': 'ap-northeast-1',
                'Asia Pacific (Sydney)': 'ap-southeast-2'
            }
            selected_region = st.selectbox(
                "AWS 리전 선택",
                options=list(region_options.keys()),
                help="Bedrock 서비스를 사용할 AWS 리전을 선택하세요"
            )
            if st.session_state.analyzer.region_name != region_options[selected_region]:
                st.session_state.analyzer.set_region(region_options[selected_region])
        
        with col3:
            analysis_mode = st.selectbox(
                "분석 모드",
                ["개선된 분석 (멀티스케일+ROI)", "계층적 분석", "단순 분석"],
                help="개선된 분석: ROI 검출 + 멀티스케일 타일링 (권장)\n계층적 분석: 2단계 분석 (개요 + 타일링)"
            )
        
        # 고급 설정
        with st.expander("🔧 고급 설정"):
            col1, col2, col3 = st.columns(3)
            with col1:
                tile_size = st.slider("타일 크기 (px)", 256, 768, 384, 32)
            with col2:
                overlap_pixels = st.slider("오버랩 크기 (px)", 64, 256, 128, 16, 
                                         help="타일 간 겹치는 픽셀 수")
            with col3:
                iou_threshold = st.slider("중복 제거 임계값", 0.1, 0.8, 0.3, 0.05, 
                                        help="낮을수록 더 많은 심볼 유지 (0.3 권장)")
        
        # 이미지 업로드
        uploaded_file = st.file_uploader(
            "P&ID 이미지 업로드",
            type=['png', 'jpg', 'jpeg'],
            help="PNG, JPG, JPEG 형식의 P&ID 도면을 업로드하세요"
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 P&ID", use_container_width=True)
            
            # 분석 실행
            if st.button("🚀 분석 시작", type="primary"):
                with st.spinner("P&ID 분석 중... 잠시만 기다려주세요."):
                    start_time = time.time()
                    
                    try:
                        if analysis_mode == "개선된 분석 (멀티스케일+ROI)":
                            # 새로운 개선된 분석 방법
                            st.info("🚀 개선된 분석 모드: ROI 검출 + 멀티스케일 타일링으로 더 정확한 심볼 검출")
                            result = st.session_state.analyzer.analyze_pnid_hierarchical(image)
                        elif analysis_mode == "계층적 분석":
                            result = st.session_state.analyzer.analyze_pnid_hierarchical(
                                image, tile_size, overlap_pixels, iou_threshold
                            )
                        else:
                            # 단순 분석 (전체 이미지 한 번에)
                            image_b64 = st.session_state.analyzer._encode_image(image)
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
      "type": "구체적 유형",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.9
    }
  ]
}

모든 bbox 좌표는 [x1, y1, x2, y2] 형식으로 정확한 픽셀 좌표를 제공하세요."""
                            response = st.session_state.analyzer._call_bedrock(prompt, image_b64)
                            
                            # JSON 파싱 시도
                            try:
                                json_start = response.find('{')
                                json_end = response.rfind('}') + 1
                                if json_start != -1 and json_end > json_start:
                                    json_str = response[json_start:json_end]
                                    parsed_result = json.loads(json_str)
                                    
                                    # 모델별 좌표 변환
                                    img_width, img_height = image.size
                                    current_model = st.session_state.analyzer.current_model
                                    
                                    # 심볼 좌표 변환
                                    for symbol in parsed_result.get("symbols", []):
                                        if 'bbox' in symbol and symbol['bbox']:
                                            symbol['bbox'] = st.session_state.analyzer._convert_1000_scale_coordinates(
                                                symbol['bbox'], img_width, img_height
                                            )
                                    
                                    print(f"[DEBUG] 단순 분석 좌표 변환 완료 - 이미지 크기: {img_width}x{img_height}")
                                    
                                    result = {
                                        "analysis_type": "단순 분석",
                                        "symbols": parsed_result.get("symbols", []),
                                        "texts": [],  # 심볼 중심이므로 빈 배열
                                        "lines": [],  # 심볼 중심이므로 빈 배열
                                        "statistics": {
                                            "total_symbols": len(parsed_result.get("symbols", [])),
                                            "total_texts": 0,
                                            "total_connections": 0
                                        }
                                    }
                                else:
                                    result = {
                                        "analysis_type": "단순 분석", 
                                        "symbols": [],
                                        "texts": [],
                                        "lines": [],
                                        "raw_response": response,
                                        "error": "JSON not found in response"
                                    }
                            except json.JSONDecodeError as e:
                                result = {
                                    "analysis_type": "단순 분석",
                                    "symbols": [],
                                    "texts": [], 
                                    "lines": [],
                                    "raw_response": response,
                                    "error": f"JSON parsing failed: {str(e)}"
                                }
                        
                        end_time = time.time()
                        result['processing_time'] = round(end_time - start_time, 2)
                        
                        st.session_state.analysis_results = result
                        
                        # 결과 저장
                        filename = save_analysis_result(result, image, selected_model)
                        st.success(f"분석 완료! ({result['processing_time']}초) - {filename}에 저장됨")
                        
                    except Exception as e:
                        st.error(f"❌ 분석 중 오류 발생: {str(e)}")
                        st.write("**오류 상세 정보:**")
                        st.code(str(e))
                        
                        # 디버그 정보 표시
                        if hasattr(st.session_state, 'analyzer') and hasattr(st.session_state.analyzer, 'debug_logs'):
                            if st.session_state.analyzer.debug_logs:
                                st.write("**마지막 API 호출 정보:**")
                                last_log = st.session_state.analyzer.debug_logs[-1]
                                st.write(f"모델: {last_log.get('model', 'Unknown')}")
                                st.text_area("응답:", last_log.get('response', 'No response')[:1000], height=200)
        
        # 분석 결과 표시
        if st.session_state.analysis_results:
            st.header("📊 분석 결과")
            result = st.session_state.analysis_results
            
            # 디버그 로그 표시
            if hasattr(st.session_state, 'analyzer') and hasattr(st.session_state.analyzer, 'debug_logs'):
                with st.expander("🐛 디버그 로그 (프롬프트 & 응답)"):
                    for i, log in enumerate(st.session_state.analyzer.debug_logs):
                        st.subheader(f"호출 {i+1} - {log['model']} ({log.get('model_id', 'N/A')})")
                        
                        # 사용된 이미지 타일 표시
                        if 'image_b64' in log:
                            try:
                                import base64
                                image_data = base64.b64decode(log['image_b64'])
                                tile_image = Image.open(io.BytesIO(image_data))
                                st.image(tile_image, caption=f"사용된 타일 {i+1}", width=300)
                            except:
                                st.write("이미지 표시 실패")
                        
                        st.text_area(
                            "프롬프트:",
                            log['prompt'],
                            height=150,
                            key=f"prompt_{i}"
                        )
                        
                        st.text_area(
                            "응답:",
                            log['response'],
                            height=200,
                            key=f"response_{i}"
                        )
                        
                        st.write(f"타임스탬프: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log['timestamp']))}")
                        st.divider()
            
            # 통계 정보
            if 'statistics' in result:
                col1, col2, col3, col4 = st.columns(4)
                stats = result['statistics']
                with col1:
                    st.metric("총 기호 수", stats.get('total_symbols', 0))
                with col2:
                    st.metric("총 텍스트 수", stats.get('total_texts', 0))
                with col3:
                    st.metric("총 연결 수", stats.get('total_connections', 0))
                with col4:
                    st.metric("처리 시간", f"{result.get('processing_time', 0)}초")
            
            # 탭으로 결과 구분
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["🔍 탐지된 기호", "🌐 지식 그래프", "📄 원시 데이터", "🐛 디버그"])
            
            with result_tab1:
                if 'symbols' in result and result['symbols']:
                    st.subheader("탐지된 기호 목록")
                    
                    # 이미지 표시 (바운딩 박스 오버레이용)
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("도면 (바운딩 박스 표시)")
                        
                        # 선택된 심볼들의 바운딩 박스를 그릴 이미지 준비
                        from PIL import Image as PILImage, ImageDraw, ImageFont
                        import numpy as np
                        
                        # PIL 이미지로 변환
                        if isinstance(image, PILImage.Image):
                            img_display = image.copy()
                        else:
                            img_display = PILImage.fromarray(np.array(image))
                        
                        # RGB 모드로 변환
                        if img_display.mode != 'RGB':
                            img_display = img_display.convert('RGB')
                        
                        # 이미지 크기 정보
                        img_width, img_height = img_display.size
                        print(f"[DEBUG] 이미지 크기: {img_width}x{img_height}")
                        
                        # 그리기 객체 생성
                        draw = ImageDraw.Draw(img_display)
                        
                        # 선택된 심볼들의 바운딩 박스 그리기
                        selected_symbols = st.session_state.get('selected_symbols', [])
                        
                        colors = [
                            (0, 255, 0),    # 초록색
                            (255, 0, 0),    # 빨간색  
                            (0, 0, 255),    # 파란색
                            (255, 255, 0),  # 노란색
                            (255, 0, 255),  # 마젠타
                            (0, 255, 255),  # 시안
                        ]
                        
                        for idx, symbol_idx in enumerate(selected_symbols):
                            if symbol_idx < len(result['symbols']):
                                symbol = result['symbols'][symbol_idx]
                                bbox = symbol.get('bbox', [0,0,0,0])
                                
                                if bbox != [0,0,0,0]:
                                    # 좌표 검증 및 클리핑
                                    x1 = max(0, min(int(bbox[0]), img_width-1))
                                    y1 = max(0, min(int(bbox[1]), img_height-1))
                                    x2 = max(x1+1, min(int(bbox[2]), img_width))
                                    y2 = max(y1+1, min(int(bbox[3]), img_height))
                                    
                                    print(f"[DEBUG] 심볼 {symbol_idx}: 원본 bbox {bbox} -> 클리핑된 bbox [{x1},{y1},{x2},{y2}]")
                                    
                                    # 색상 선택 (순환)
                                    color = colors[idx % len(colors)]
                                    
                                    # 바운딩 박스 그리기 (두꺼운 선)
                                    for i in range(5):  # 5픽셀 두께
                                        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color, width=1)
                                    
                                    # 라벨 텍스트
                                    label = f"{symbol.get('class', 'Unknown')[:10]}"
                                    
                                    # 라벨 배경 그리기
                                    try:
                                        font = ImageFont.load_default()
                                        bbox_text = draw.textbbox((0, 0), label, font=font)
                                        text_width = bbox_text[2] - bbox_text[0]
                                        text_height = bbox_text[3] - bbox_text[1]
                                    except:
                                        text_width, text_height = len(label) * 8, 15
                                    
                                    # 배경 사각형
                                    draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)
                                    
                                    # 라벨 텍스트 (흰색)
                                    draw.text((x1+5, y1-text_height-2), label, fill=(255, 255, 255))
                                    
                                    # 좌표 정보 표시
                                    coord_text = f"({x1},{y1})"
                                    draw.text((x1, y2+5), coord_text, fill=color)
                        
                        # 이미지 표시
                        st.image(img_display, caption=f"P&ID 도면 (크기: {img_width}x{img_height}, 선택된 부품: {len(selected_symbols)}개)", use_column_width=True)
                        
                        # 좌표 정보 표시
                        if selected_symbols:
                            st.write("**선택된 부품 좌표 정보:**")
                            for idx, symbol_idx in enumerate(selected_symbols):
                                if symbol_idx < len(result['symbols']):
                                    symbol = result['symbols'][symbol_idx]
                                    bbox = symbol.get('bbox', [0,0,0,0])
                                    color_name = ["초록", "빨강", "파랑", "노랑", "마젠타", "시안"][idx % 6]
                                    st.write(f"- {color_name}: {symbol.get('class', 'Unknown')} [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                    
                    with col2:
                        st.subheader("부품 목록")
                        
                        # 선택된 심볼 상태 초기화
                        if 'selected_symbols' not in st.session_state:
                            st.session_state.selected_symbols = []
                        
                        # 모든 심볼 선택/해제 버튼
                        col_all1, col_all2 = st.columns(2)
                        with col_all1:
                            if st.button("전체 선택"):
                                st.session_state.selected_symbols = list(range(len(result['symbols'])))
                                st.rerun()
                        with col_all2:
                            if st.button("전체 해제"):
                                st.session_state.selected_symbols = []
                                st.rerun()
                        
                        # 각 심볼별 체크박스와 정보
                        for i, symbol in enumerate(result['symbols']):
                            bbox = symbol.get('bbox', [0,0,0,0])
                            coord_info = f"({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})" if bbox != [0,0,0,0] else "좌표 없음"
                            
                            # 체크박스
                            is_selected = st.checkbox(
                                f"{symbol.get('class', 'Unknown')} - {symbol.get('tag_id', f'Symbol_{i}')}",
                                value=i in st.session_state.selected_symbols,
                                key=f"symbol_check_{i}"
                            )
                            
                            # 선택 상태 업데이트
                            if is_selected and i not in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.append(i)
                            elif not is_selected and i in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.remove(i)
                            
                            # 심볼 상세 정보 (접을 수 있는 형태)
                            with st.expander(f"상세 정보 - {coord_info}", expanded=False):
                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.write(f"**클래스**: {symbol.get('class', 'Unknown')}")
                                    st.write(f"**유형**: {symbol.get('type', 'N/A')}")
                                    st.write(f"**태그 ID**: {symbol.get('tag_id', 'N/A')}")
                                    st.write(f"**신뢰도**: {symbol.get('confidence', 0):.2f}")
                                    if 'source' in symbol:
                                        st.write(f"**출처**: {symbol['source']}")
                                with col_info2:
                                    st.write(f"**좌표 (x1, y1)**: ({bbox[0]}, {bbox[1]})")
                                    st.write(f"**좌표 (x2, y2)**: ({bbox[2]}, {bbox[3]})")
                                    st.write(f"**크기**: {bbox[2]-bbox[0]} × {bbox[3]-bbox[1]} px")
                                    if bbox != [0,0,0,0]:
                                        st.write(f"**중심점**: ({(bbox[0]+bbox[2])//2}, {(bbox[1]+bbox[3])//2})")
                else:
                    st.info("탐지된 기호가 없습니다.")
            
            with result_tab2:
                if 'knowledge_graph' in result:
                    st.subheader("지식 그래프 시각화")
                    visualize_knowledge_graph(result['knowledge_graph'])
                else:
                    st.info("지식 그래프 데이터가 없습니다.")
            
            with result_tab3:
                st.subheader("원시 분석 데이터")
                st.json(result)
            
            with result_tab4:
                st.subheader("디버그 정보")
                if 'raw_response' in result:
                    st.text_area("AI 모델 원시 응답", result['raw_response'], height=200)
                if 'error' in result:
                    st.error(f"파싱 오류: {result['error']}")
                
                # 통계 정보
                st.write("**분석 통계:**")
                st.write(f"- 분석 유형: {result.get('analysis_type', 'Unknown')}")
                st.write(f"- 처리 시간: {result.get('processing_time', 'N/A')}초")
                st.write(f"- 총 타일 수: {result.get('total_tiles', 'N/A')}")
                
                if 'statistics' in result:
                    stats = result['statistics']
                    st.write(f"- 탐지된 기호: {stats.get('total_symbols', 0)}개")
                    st.write(f"- 추출된 텍스트: {stats.get('total_texts', 0)}개")
                    st.write(f"- 연결 관계: {stats.get('total_connections', 0)}개")
    
    with tab2:
        st.header("📁 저장된 분석 결과")
        
        saved_results = load_saved_results()
        if saved_results:
            for result in saved_results:
                with st.expander(f"📄 {result['filename']} - {result['timestamp']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**모델**: {result['model']}")
                    with col2:
                        st.write(f"**기호 수**: {result['total_symbols']}")
                    with col3:
                        if st.button("불러오기", key=f"load_{result['filename']}"):
                            try:
                                with open(f"saved_results/{result['filename']}", 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    st.session_state.analysis_results = data['analysis_result']
                                    st.success("결과를 불러왔습니다!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"파일 로드 실패: {str(e)}")
        else:
            st.info("저장된 분석 결과가 없습니다.")
    
    with tab3:
        st.header("⚙️ 시스템 설정")
        
        st.subheader("AWS 설정")
        st.info("AWS 자격 증명이 올바르게 설정되어 있는지 확인하세요.")
        st.code("aws configure", language="bash")
        
        st.subheader("모델 정보")
        model_info = {
            "Claude 3.5 Sonnet": "최고 성능, 균형잡힌 비용",
            "Claude 3.5 Haiku": "빠른 처리, 저비용",
            "Claude 3 Sonnet": "안정적 성능",
            "Amazon Nova Pro": "멀티모달 우수, 비용 효율적",
            "Amazon Nova Lite": "최고 속도, 간단한 분석"
        }
        
        for model, desc in model_info.items():
            st.write(f"**{model}**: {desc}")
        
        st.subheader("성능 최적화 팁")
        st.markdown("""
        - **타일 크기**: 512px가 일반적으로 최적
        - **겹침 비율**: 0.5 (50%)가 권장값
        - **계층적 분석**: 복잡한 도면에 권장
        - **단순 분석**: 간단한 도면이나 빠른 확인용
        """)
        
        st.subheader("API 제한 및 오류 대응")
        st.markdown("""
        **ThrottlingException 오류 시:**
        - 자동 재시도 로직이 작동합니다 (최대 5회)
        - 지수 백오프로 대기 시간이 증가합니다
        - 계속 오류 발생 시 몇 분 후 다시 시도하세요
        
        **권장 사항:**
        - 대형 이미지는 작은 타일 크기 사용
        - 여러 분석을 연속으로 실행하지 마세요
        - Claude 3 Haiku 모델 사용으로 비용 절약
        """)

if __name__ == "__main__":
    main()

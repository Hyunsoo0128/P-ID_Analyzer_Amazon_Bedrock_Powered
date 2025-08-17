# 🔧 P&ID Analyzer - Amazon Bedrock Powered

A high-performance P&ID (Piping and Instrumentation Diagram) analysis solution powered by Amazon Bedrock foundation models with hierarchical pipeline architecture.

## ✨ Key Features

### 🤖 Multi-Model Support
- **Claude Models**: 3.5 Sonnet, 3.5 Haiku, 3 Sonnet
- **Amazon Nova Models**: Nova Pro, Nova Lite, Nova Premier
- **Auto-compatibility**: Automatic API format conversion between models

### 🏗️ Hierarchical Pipeline Architecture
1. **Adaptive Tiling**: Intelligent image segmentation based on regions of interest
2. **Parallel Analysis**: Independent parallel processing of each tile
3. **Knowledge Graph Synthesis**: Consolidating distributed results into unified graph
4. **HITL Validation**: Human expert validation and continuous improvement

### 📊 Advanced Analysis Capabilities
- **Symbol Recognition**: ISA S5.1 standard-based P&ID symbol detection
- **Text Extraction**: Tag ID and specification information OCR
- **Connection Inference**: Piping and signal line relationship analysis
- **Deduplication**: NMS algorithm-based duplicate detection and merging

## 🚀 Quick Start

### Prerequisites
- AWS Account with Bedrock access
- Python 3.8+
- Required AWS permissions: `bedrock:InvokeModel`

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/pnid-analyzer.git
cd pnid-analyzer

# Create virtual environment
python3 -m venv pnid_env
source pnid_env/bin/activate  # Linux/Mac
# pnid_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### AWS Configuration
```bash
# Configure AWS CLI (Bedrock access required)
aws configure

# Required permissions:
# - bedrock:InvokeModel
# - bedrock:InvokeModelWithResponseStream
```

### Usage

#### Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

#### CLI Usage
```python
from pnid_analyzer import PNIDAnalyzer
from PIL import Image

# Initialize analyzer
analyzer = PNIDAnalyzer()
analyzer.set_model("claude-3-5-sonnet")

# Load image
image = Image.open("sample_pnid.png")

# Run hierarchical analysis
result = analyzer.analyze_pnid_hierarchical(
    image, 
    tile_size=512, 
    overlap_ratio=0.5
)

print(f"Detected symbols: {len(result['symbols'])}")
print(f"Connections: {result['statistics']['total_connections']}")
```

## 📁 Project Structure

```
pnid-analyzer/
├── pnid_analyzer.py          # Core analysis engine
├── streamlit_app.py          # Web interface with HITL validation
├── example_usage.py          # CLI usage examples
├── requirements.txt          # Dependencies
├── saved_results/           # Analysis results storage
└── README.md               # Project documentation
```

## 🎯 Analysis Modes

### Enhanced Analysis (Recommended)
- Multi-scale adaptive tiling (256px, 384px, 512px)
- ROI-based interest region detection
- Adaptive overlap ratios based on feature density
- Optimal for complex diagrams with varying symbol sizes

### Hierarchical Analysis
- Two-stage processing pipeline
- Coarse-to-fine analysis approach
- Balanced performance and accuracy

### Simple Analysis
- Single-pass full image analysis
- Fast processing for simple diagrams

## 📊 Output Format

```json
{
  "analysis_type": "Enhanced P&ID Analysis",
  "total_tiles": 49,
  "processing_time": 45.2,
  "symbols": [
    {
      "id": "symbol_1",
      "class": "Centrifugal Pump",
      "bbox": [150, 200, 200, 250],
      "confidence": 0.95,
      "tag_id": "P-101"
    }
  ],
  "texts": [
    {
      "id": "text_1",
      "content": "P-101",
      "bbox": [155, 180, 195, 195],
      "associated_symbol": "symbol_1"
    }
  ],
  "lines": [
    {
      "id": "line_1",
      "type": "process_major",
      "path": [[200, 225], [300, 225]],
      "connections": ["symbol_1", "symbol_2"]
    }
  ],
  "knowledge_graph": {
    "nodes": [...],
    "edges": [...]
  },
  "statistics": {
    "total_symbols": 45,
    "total_texts": 120,
    "total_connections": 38
  }
}
```

## 🔧 Technical Architecture

### Core Components

1. **Adaptive Tiling Engine**
   - Low-resolution ROI mapping
   - High-resolution overlapping tile generation
   - Cost-optimized tile planning

2. **Multimodal Analysis Engine**
   - ISA S5.1 standard-based prompts
   - Few-shot learning implementation
   - Structured JSON output

3. **Knowledge Graph Synthesis**
   - NMS-based deduplication
   - Connection relationship inference
   - NetworkX graph structure

4. **HITL Validation System**
   - Interactive visualization
   - Real-time modification capabilities
   - Feedback loop construction

### Model Performance Characteristics

| Model | Vision Capability | Reasoning | Cost | Recommended Use Case |
|-------|------------------|-----------|------|---------------------|
| Claude 3.5 Sonnet | Excellent | High | Medium | Main tile analysis, complex symbol recognition |
| Claude 3.5 Haiku | Good | Medium | Low | ROI mapping, simple analysis |
| Nova Pro | Good | High | Low | Cost-effective multimodal analysis |
| Nova Lite | Fair | Medium | Very Low | Fast processing, simple diagrams |

## 📈 Performance Benchmarks

### Processing Time
- **Small diagrams** (1000x800): ~30 seconds
- **Medium diagrams** (2000x1500): ~60 seconds  
- **Large diagrams** (4000x3000): ~120 seconds

### Optimization Strategies
- **Default Strategy**: 512x512 pixels, 50% overlap
- **Adaptive Strategy**: ROI-based selective high-resolution processing
- **Cost Reduction**: Automatic empty region exclusion, hierarchical model usage

## 🎯 Use Cases

- Chemical plant design review
- Petrochemical process analysis
- Pharmaceutical facility management
- Power plant system analysis
- Water treatment facility management
- Building MEP drawing analysis

## 🔍 Key Technical Innovations

### ✅ Small Object Detection
- **Challenge**: Missing small symbols in full image analysis
- **Solution**: Adaptive tiling preserves resolution

### ✅ Connection Relationship Inference
- **Challenge**: Complex piping connection identification
- **Solution**: Line tracing + graph-based connection inference

### ✅ Model Compatibility
- **Challenge**: API format differences between Claude and Nova models
- **Solution**: Automatic message format conversion

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues and pull requests.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black pnid_analyzer.py streamlit_app.py
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 References

- [ISA S5.1 Standard](https://www.isa.org/standards-and-publications/isa-standards/isa-standards-committees/isa5)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [DEXPI Standard](https://dexpi.org/)

## 👨‍💻 Author

**김현수 (Hyunsoo Kim)**  
Senior GenAI Specialist Solutions Architect  
Amazon Web Services (AWS)

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: August 2025

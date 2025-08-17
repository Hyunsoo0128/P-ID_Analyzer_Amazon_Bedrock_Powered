#!/bin/bash

# GitHub 저장소 설정 스크립트
# 사용법: ./setup_github.sh

echo "🔧 P&ID Analyzer GitHub 저장소 설정"
echo "======================================"

# Git 초기화
if [ ! -d ".git" ]; then
    echo "📁 Git 저장소 초기화..."
    git init
else
    echo "✅ Git 저장소가 이미 존재합니다."
fi

# .gitignore 적용
echo "📝 .gitignore 적용..."
git add .gitignore

# 모든 파일 추가
echo "📦 프로젝트 파일 추가..."
git add .

# 초기 커밋
echo "💾 초기 커밋 생성..."
git commit -m "Initial commit: P&ID Analyzer - Amazon Bedrock Powered

- Core analysis engine with hierarchical pipeline
- Multi-model support (Claude, Nova)
- Streamlit web interface
- Enhanced analysis with ROI detection
- Complete documentation and tests

Author: 김현수 (Hyunsoo Kim), Senior GenAI Specialist SA, AWS"

echo ""
echo "✅ 로컬 Git 저장소 설정 완료!"
echo ""
echo "다음 단계:"
echo "1. GitHub에서 새 저장소 생성 (예: pnid-analyzer)"
echo "2. 다음 명령어로 원격 저장소 연결:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/pnid-analyzer.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🚀 GitHub 공유 준비 완료!"

📖 README.md
프로젝트 개요

이 프로젝트는 점(dot) 기반 문자 인식 시스템을 구축하기 위한 코드와 데이터셋을 포함합니다.

데이터 생성기 (dot_font_generator.py) : 점 폰트 문자 이미지와 CSV 생성

분석/전처리 UI (dot_font_analyzer.py) : ROI/threshold/clustering/grid 변환

생성된 데이터(chars.csv, images.csv)를 바탕으로 인식률 평가 및 학습 준비

추후 딥러닝 학습으로 확장 가능

📂 폴더 구조
project_root/
├─ manual_dot_image/ # 실제 학습/분석용 데이터
│ ├─ images/ # 점 폰트 이미지 (랜덤 생성 결과)
│ ├─ chars.csv # 문자 단위 라벨 (정답 데이터)
│ └─ images.csv # 이미지 단위 라벨 (정답 데이터)
├─ dot_font_generator.py # 점 폰트 이미지 + CSV 생성기
├─ dot_font_analyzer.py # UI + 클러스터링 + 그리드 변환 프로그램
└─ README.md

📝 주요 파일 설명

dot_font_generator.py
무작위 점 폰트 문자를 생성하여 manual_dot_image/images/ 에 저장합니다.
동시에 chars.csv, images.csv 정답 데이터를 생성합니다.

dot_font_analyzer.py
Tkinter UI를 제공하여 이미지 분석 및 인식 파이프라인을 수행합니다.

분석 시작: ROI/threshold/clustering/grid 설정 후 결과 저장

학습 시작: chars.csv와 비교하여 문자 인식률 평가 (90% 이상 목표)

manual_dot_image/

images/: 생성된 문자 이미지

chars.csv: 문자 단위 정답 데이터

images.csv: 이미지 단위 정답 데이터

🚀 실행 예시

점 폰트 데이터 생성하기

# 랜덤 문자 이미지와 CSV 데이터 생성

python dot_font_generator.py

실행 후 manual_dot_image/ 폴더 안에:

images/ # 생성된 이미지
chars.csv # 문자 단위 라벨
images.csv # 이미지 단위 라벨

이 생성됩니다.

UI 실행 후 분석/학습하기

# UI 실행

python dot_font_analyzer.py

ROI/Threshold/Clustering 파라미터를 조정하면서 이미지 확인

분석 시작 버튼 → 클러스터링 결과와 그리드 저장

학습 시작 버튼 → manual_dot_image/chars.csv 불러와 인식률 평가

🔮 향후 확장

dot_font_trainer.py : 딥러닝 기반 학습 모듈 (PyTorch/TensorFlow)

dot_font_evaluator.py : 학습된 모델 평가 모듈

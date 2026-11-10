# 중고거래 서비스 데이터 구축 및 사용자 행동 분석

중고거래 판매글의 `조회 → 찜 → 재조회 → 채팅 → 거래` 행동을 분석해 채팅 전환이 정체되는 구간과 관련 요인을 살펴본 데이터 분석 포트폴리오입니다.

> 본 프로젝트의 데이터는 실제 사용자 데이터가 아닌 규칙 기반 합성 데이터입니다.

## 핵심 결과

| 분석 항목 | 결과 |
|---|---:|
| 전체 사용자-판매글 여정 | 36,360건 |
| 전체 채팅 | 2,620건 |
| 찜 없이 직접 채팅 | 1,769건 (67.52%) |
| 찜 후 채팅 | 851건 (32.48%) |
| 직접 채팅 거래 시작률 | 26.85% |
| 찜 후 채팅 거래 시작률 | 26.32% |
| 찜 후 재조회 집단 중 채팅 미전환 | 732건 (83.75%) |
| 찜 후 재조회 집단 중 채팅 전환 | 142건 (16.25%) |
| 재조회 집단의 채팅 전환율 | 가격 제안 불가능 13.45% / 가능 19.16% |

- 직접 채팅은 전체 채팅의 67.52%였으며, 거래 시작률도 찜 후 채팅과 거의 같았습니다.
- 찜한 판매글을 다시 조회해도 83.75%는 채팅으로 이어지지 않았습니다.
- 재조회 집단에서는 가격 제안 가능 판매글의 채팅 전환율이 불가능 판매글보다 5.71%p 높았습니다.
- 로지스틱 회귀와 랜덤 포레스트의 ROC-AUC는 각각 0.5498, 0.5444였습니다. 현재 변수만으로 개인의 전환 여부를 정확히 예측하기는 어려웠습니다.

## 분석 흐름

1. 사용자-판매글 단위로 행동 여정을 재구성했습니다.
2. 직접 채팅과 찜 후 채팅의 규모 및 거래 성과를 비교했습니다.
3. `찜 → 재조회` 이후 채팅 전환/미전환 집단을 비교했습니다.
4. 반복 조회, 동일 지역, 찜 전 가격 인하, 가격 제안 가능 여부로 채팅 전환 모델을 학습했습니다.

## 데이터 규모

| 테이블 | 행 수 |
|---|---:|
| users | 1,000 |
| listings | 1,500 |
| product_views | 45,367 |
| favorites | 5,519 |
| chatrooms | 2,733 |
| transactions | 699 |
| price_history | 2,204 |

관찰 기간은 2025-12-01부터 2026-02-28까지 90일입니다. 데이터 생성 시 PK·FK 관계와 행동 시간 순서를 검증했습니다.

## 저장소 구성

```text
carrot/
├── config/                 # 합성 데이터 생성 설정
├── generated/              # 합성 데이터와 실행 기록
│   └── csv/                # 7개 테이블 CSV
├── notebooks/
│   └── DA_final.ipynb      # 최종 분석 노트북
├── prompts/                # LLM 자산 생성 프롬프트
├── reports/                # 데이터 생성·검증 보고서
├── sql/                    # 스키마, 제약조건, 분석 View
├── src/                    # 데이터 생성·검증·MySQL 적재 코드
└── tableau/                # Tableau 대시보드용 CSV
```

## 실행 방법

### 1. Python 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
```

`.env`에 MySQL 접속 정보를 입력합니다. 실제 `.env`는 Git에 포함되지 않습니다.

### 3. 데이터 생성 및 검증

MySQL 적재 없이 CSV와 검증 보고서만 생성하려면 다음 명령을 실행합니다.

```bash
python src/run_pipeline.py --skip-mysql
```

MySQL 적재까지 실행하려면 `.env` 설정과 빈 테이블 7개가 필요합니다.

```bash
python src/run_pipeline.py
```

> 주의: [`sql/schema.sql`](sql/schema.sql)은 기존 `carrot_chat` 데이터베이스를 삭제하고 다시 만드는 구문으로 시작합니다. 기존 DB가 있다면 실행하지 마세요.

### 4. 분석 View 생성 및 노트북 실행

```bash
mysql -u 사용자명 -p carrot_chat < sql/listing_funnel.sql
jupyter notebook notebooks/DA_final.ipynb
```

### 5. Tableau 대시보드 데이터 생성

```bash
python src/export_tableau_data.py
```

실행하면 `tableau/tableau_dashboard_data.csv`가 생성됩니다.

## 주요 파일

- [최종 분석 노트북](notebooks/DA_final.ipynb)
- [데이터 생성 보고서](reports/generation_report.json)
- [분석 View SQL](sql/listing_funnel.sql)
- [데이터 생성 파이프라인](src/run_pipeline.py)
- [Tableau 대시보드 데이터](tableau/tableau_dashboard_data.csv)
- [Tableau 데이터 생성 코드](src/export_tableau_data.py)

## 해석 시 유의사항

- 합성 데이터이므로 실제 서비스 사용자의 인과관계를 증명하지 않습니다.
- 모델 성능이 기준선에 가까워, 결과는 개인 예측보다 퍼널 진단과 가설 도출에 활용했습니다.
- 조회·찜·채팅의 일부 시간 간격은 생성 규칙의 영향을 받을 수 있어 핵심 결론에서 제외했습니다.

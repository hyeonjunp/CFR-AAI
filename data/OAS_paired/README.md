# OAS Paired — 항체 CDR 데이터셋

OAS(Observed Antibody Space) Paired 전체를 받아 heavy/light 로 나누고
IMGT 정의 CDR/FR 을 추출한 결과.

## 소스
- https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/ (21 studies / 610 data units)
- 원본 압축 3.3GB, 유니크 서열 3,087,576 paired
- CDR/FR 은 OAS 의 `fwr*_aa` / `cdr*_aa` 컬럼(ANARCI IMGT 넘버링)을 그대로 사용
- OAS 에 `fwr4` 컬럼이 없어 **FR4 는 `sequence_alignment_aa` 의 잔여 접미사로 복원**

## 파이프라인
```bash
scripts/01_download.sh 10          # 원본 다운로드 (재개 가능)
scripts/02_extract_cdr.py          # heavy/light 분리 + CDR/FR 추출
scripts/03_filter_locus.py         # locus 오배정 행 제거 (02 에 이미 반영됨, 기존 산출물 보정용)
scripts/run_all.sh                 # 다운로드→CDR 추출→클러스터링/split→리포트 전체
```

## 산출물
| 경로 | 내용 |
|---|---|
| `raw/<Study>/*.csv.gz` | OAS 원본 |
| `processed/heavy/heavy.csv.gz` | heavy 체인 (long) |
| `processed/light/light.csv.gz` | light 체인 (long, K/L 은 `locus` 컬럼) |
| `processed/paired/paired.csv.gz` | H/L 한 행 (`pair_id` 로 연결) |
| `processed/unit_summary.csv` | 데이터유닛별 통계 |

## 필터
- `productive == T`, `stop_codon != T`, `vj_in_frame != F`
- `locus` 가 체인과 일치 (heavy=H, light=K/L)
- FR1..CDR3 이 `sequence_alignment_aa` 의 접두사여야 함 (FR4 복원 가능 조건)
- 서열은 표준 20종 아미노산만 (X, `*` 포함 행 제거)
- `quality_flag`: FR4 8잔기 미만 또는 CDR3 길이 1~40 밖이면 `check` (제거 안 함, 표시만)

## 컬럼 규약
`CDR{n}_start` 는 0-based, `CDR{n}_end` 는 exclusive.
즉 `sequence[CDR1_start:CDR1_end] == CDR1`. `CDR_prediction/Dataset` 과 동일한 규약.
`mask` 는 `sequence` 와 같은 길이의 0/1 문자열 (CDR 위치가 1).

## ⚠ 읽을 때 주의
아미노산 서열 **"NA"(Asn-Ala)** 를 pandas 가 결측치로 바꿔버린다. 반드시:
```python
pd.read_csv(path, keep_default_na=False, na_values=[])
```
헬퍼 제공:
```python
from scripts.load import load_oas
df = load_oas("heavy")
```

## 편향 주의
- `Jaffe_2022` 한 스터디가 전체의 45% (donor 4명). study/subject 단위 split 권장.
- Naive B세포 비중이 큼 → SHM 거의 없어 CDR1/2 가 germline 에 가까움. `btype` 으로 층화 가능.

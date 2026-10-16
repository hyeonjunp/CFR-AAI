# data/il6 데이터셋

`il6_aai_dataset.csv` (AVIDa-hIL6 원본, 573,891 pair)를 AVIDa_hIL6 폴더와 동일한 구조로 변환하고,
ANARCI IMGT 넘버링으로 CDR을 추출한 데이터셋.

## 재생성

```bash
conda activate AbTLAS          # ANARCI 1.3 + hmmer + sklearn 이 설치된 유일한 env
cd Ab_Ag_interaction/data/il6

python build_il6_dataset.py --ncpu 16    # antibody/antigen/pair + ag_unseen 분할  (~32초)
python extract_imgt_cdr.py  --ncpu 16    # antibody_imgt_cdr.csv                  (~26초)
```

`il6_aai_dataset.csv` 하나만 있으면 나머지는 전부 재생성된다.

## 파일

| 파일 | 행 수 | 컬럼 |
|---|---|---|
| `antibody.csv` | 37,923 | `ab_id`, `ab_seq` |
| `antigen.csv` | 31 | `ag_id`, `ag_seq`, `Ag_label` |
| `ab_ag_pair.csv` | 566,554 | `label`, `Ag_label`, `ab_id`, `ag_id` |
| `antibody_imgt_cdr.csv` | 37,923 | `ab_id`, `ab_seq`, `chain_type`, `fr1`~`fr4`, `cdr1`~`cdr3`, `cdr_mask`, `domain_start`, `domain_end`, `status` |
| `ag_unseen/{train,val,unseen}_seed_{0..4}.csv` | seed별 상이 | `select_index`, `label`, `Ag_label`, `ab_id`, `ag_id` |

- `ab_id`는 `antibody.csv`의 행 번호(0..37922)와 일치한다.
- `select_index`는 `ab_ag_pair.csv`의 행 인덱스다.
- VHH 길이 95/124/151 (min/median/max), 항원은 전부 183aa.
- label 분포: 음성 546,057 / 양성 20,497 (**양성률 3.62%**).

## 전처리 파이프라인

### 1. 태그 제거

원본 서열은 발현 컨스트럭트 전체라 가변도메인 바깥에 상수 구간이 붙어 있다.

```
VHH  : MKYLLPTAAAGLLLLAAQPAMA │ QVQLQESGGG ... QVTVSS │ HHHHHH
       └─ PelB signal peptide (22aa)                  └─ His6
IL-6 : MNSFSTSAFGPVAFSLGLLLVLPAAFPAP │ VPPGEDSKD ... RALRQM │ HHHHHH
       └─ native signal peptide (29aa)                      └─ His6
```

모든 서열에 동일하게 붙어 있어 결합 여부와 무관하고 ESM2 입력 길이만 잡아먹으므로 제거한다.

단순 `str.removeprefix()`로는 부족하다 — VHH 38,599개 중 527개(1.4%)는 signal peptide 안에 점돌연변이가
있어(`MKYLLPTAAAGLLLLAAQPA[L]A` 등) 문자열이 정확히 일치하지 않는다. 앞 22aa를 canonical PelB와
hamming 거리로 비교해 3 이하면 잘라낸다(관측된 최대 mismatch는 1). His-tag는 C말단 H 연속 구간으로 찾는다.

태그를 떼면 signal peptide 변이로만 달랐던 VHH들이 같은 서열로 합쳐진다: **38,599 → 38,109 unique**.

`--keep_tags`로 끌 수 있다.

### 2. CDR 추출 불가 서열 제거

ANARCI IMGT 넘버링으로 CDR을 뽑을 수 없는 VHH **186개(0.49%)**를 제거한다.

| 유형 | 개수 | 내용 |
|---|---|---|
| `no_domain` | 84 | 가변도메인 자체가 안 잡힘 |
| `missing_cdr` | 102 | 도메인은 잡혔으나 CDR 구간이 빔 (cdr1 76 / cdr2 65 / cdr3 26) |

`no_domain` 84개 중 **58개는 길이가 정상(121~126aa)인데도 실패**한다. 이들은 전부 동일하게 손상된
N말단 블록을 공유하는 프레임시프트 아티팩트다:

```
정상   : QVQLQESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISWSGGSTYYADSVKGRFTISRDNANNTVY
망가짐 : QVQLQIHRVLGVVSGDGESALHGVCIICAITTAINIGDPLEPLSWSLADPAHSIYTEGESRGSTGESQRPPRLHQASPRL
         └── QVQLQ 5잔기까지만 같고 그 뒤 ~75잔기가 완전히 다른 서열 ──┘
```

나머지 26개는 중간이 결실된 truncated 서열(72~126aa)이다.

주의할 점: 이들이 잃은 것은 **CDR3가 아니라 FR1/CDR1/FR2/CDR2**다. `YYC...WGQG` 모티프 기준으로
CDR3 구간은 프레임시프트형 54/58, 잘린 서열형 17/26에서 온전하다. ANARCI는 가변도메인 전체를 HMM으로
정렬하므로 앞쪽 절반이 germline과 안 맞으면 뒤쪽 정상 구간까지 함께 버린다. CDR3만 정규식으로 건질 수는
있으나 FR1/CDR1/CDR2는 복구 불가이고 IMGT 경계가 아닌 모티프 근사가 섞이므로 통째로 제거한다.

제거는 `ab_id` 부여 **전에** 이뤄져 id가 0..N-1로 연속된다. `--keep_broken`으로 끌 수 있다.

### 3. 참조 끊긴 pair 제거

제거된 VHH를 참조하는 pair **2,533개**(label 0이 2,511, label 1이 22)를 함께 버린다.
573,891 → 571,358 rows.

### 4. 중복 pair 정리

태그 제거로 VHH가 합쳐지면서 같은 `(ab_id, ag_id)` 조합이 여러 번 나타난다(중복 조합 1,939개).
**반드시 train/val 분할 전에** 정리해야 한다 — 그대로 두면 동일 샘플이 train과 val 양쪽에 들어가
val 성능이 부풀려진다.

- label 일치 중복 → 하나만 남긴다 (정보 손실 없음). −4,511 rows
- label 충돌 조합(80개) → 조합 전체를 버린다. −293 rows

  충돌의 절반 이상이 정확히 동점(label 평균 0.5)이라 다수결이 성립하지 않고, 나머지도 2~3 rows
  근거뿐이라 신뢰할 수 없다.

571,358 → **566,554 rows**. `--keep_dupes`로 끌 수 있다.

## 제거 누계

| 단계 | VHH | pair |
|---|---:|---:|
| 원본 | 38,599 | 573,891 |
| 태그 제거로 중복 병합 | 38,109 (−490) | 573,891 |
| CDR 추출 불가 제거 | **37,923** (−186) | 571,358 (−2,533) |
| 중복/충돌 pair 정리 | 37,923 | **566,554** (−4,804) |
| **최종** | **37,923** (−676, 1.75%) | **566,554** (−7,337, 1.28%) |

## IMGT CDR 추출

`extract_imgt_cdr.py` — ANARCI 1.3으로 IMGT 넘버링 후 구간별로 자른다.

```
FR1 1-26 │ CDR1 27-38 │ FR2 39-55 │ CDR2 56-65 │ FR3 66-104 │ CDR3 105-117 │ FR4 118-128
```

IMGT는 gap을 넣어 길이를 맞추는 스킴이라 넘버링 결과에 `-`가 섞인다. 실제 서열만 필요하므로 gap은
버리고 residue만 모은다. insertion code(111A, 112A ...)는 CDR3 길이 확장용이라 그대로 포함된다.

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `-i` / `-o` | `antibody.csv` / `antibody_imgt_cdr.csv` | 입출력 경로 |
| `--id_col` / `--seq_col` | `ab_id` / `ab_seq` | 컬럼명 |
| `--ncpu` | CPU 전체 | ANARCI 병렬도 |
| `--flank N` | 0 | CDR 양옆 N잔기 확장. AVIDa_hIL6의 `antibody.csv`가 이런 확장 정의를 쓴다 |
| `--species` | 전체 | ANARCI allowed_species (예: `alpaca,human`) |
| `--drop_failed` | off | `no_domain` / `missing_cdr` 제외 |

`cdr_mask`는 `ab_seq`와 같은 길이의 0/1 문자열로 CDR 위치를 표시한다.

추출 결과 (정제된 antibody.csv 기준 **37,923 / 37,923, 100%** 성공):

| | 길이 min/median/max | unique |
|---|---|---:|
| cdr1 | 1 / 8 / 17 | 11,254 |
| cdr2 | 2 / 8 / 18 | 12,085 |
| cdr3 | 2 / 17 / 36 | 18,705 |

`build_il6_dataset.py`는 이 스크립트의 `extract_cdrs()`를 import해서 쓴다 — CDR 정의가 두 곳으로
갈라지지 않게 하기 위함.

## ag_unseen 분할

기존 `AVIDa_hIL6/ag_unseen` 파일을 분석해 역산한 로직:

- WT 항원(`IL-6_WTs`)은 항상 seen(train/val)에 포함
- 나머지 30개 mutant 항원을 seed별로 반씩(15/15) seen/unseen으로 무작위 분할
- seen 항원의 pair를 `train_test_split(test_size=0.1, random_state=seed)`로 train/val 분할
- unseen 항원의 pair는 그대로 `unseen_seed_N.csv` (원본 순서 유지, shuffle 없음)

| seed | train | val | unseen |
|---|---:|---:|---:|
| 0 | 253,792 | 28,200 | 284,562 |
| 1 | 267,202 | 29,690 | 269,662 |
| 2 | 256,048 | 28,450 | 282,056 |
| 3 | 239,989 | 26,666 | 299,899 |
| 4 | 270,986 | 30,110 | 265,458 |

## 검증

재생성 후 확인한 항목 (전부 통과):

- `ab_id`가 0..37922로 연속
- `ab_ag_pair.csv` → `antibody.csv` / `antigen.csv` 참조 무결성, dangling 0
- 중복 `(ab_id, ag_id)` 0
- `antibody_imgt_cdr.csv`가 `antibody.csv`와 행 순서·서열 일치, 전 행 `status == ok`
- 5개 seed 전부: train/val/unseen이 pair 전체를 겹침 없이 덮음
- `select_index`가 `ab_ag_pair.csv` 행과 정확히 일치
- train↔unseen 항원 누수 없음

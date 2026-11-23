# ROCF VLM 합성 데이터

[ROCF VLM 챌린지](../rocf_vlm.md)에 쓰는 **연습·검증용 합성 디지털 펜 기록과 학습 전용 뇌지표**를 만드는 코드, 정적 기준선 파이프라인, 규격 검증기, 예제 노트북입니다. 실제 사람의 임상·뇌영상 기록이 아니며, 블라인드 테스트 정답(`test_labels`, `test_diagnoses`)과 생성 설정이 `data/ground_truth.json`에 따로 저장됩니다.

## 들어 있는 파일

| 파일 | 내용 |
| --- | --- |
| `rocf_synth.py` | 18개 ROCF 도형 요소 기하 정의, 5 Hz 펜 궤적·학습 전용 뇌지표 생성, 정적 흑백 이미지·과정 진행 패널 렌더링(`render_process_panel`), 획 순서·시각 셔플(`shuffle_trial_strokes`), 진단군 내 층화 AUROC 계산 함수 모음 |
| `generate_synthetic_rocf.py` | 참가자 분할 표(`participants.csv`), 학습 전용 뇌·임상 특권 표(`train_privileged.csv`), `Copy`·`Delayed Recall` 그림 및 5 Hz 펜 기록 생성 프로그램 (`--seed`, `--out`, `--null-effect`, `--distribution-shift` 지원) |
| `starter_pipeline.py` | 실행 규격 확인용 정적 흑백 그림 전용 최소 기준선(진단군 교란 착시와 진단군 내 AUROC ≈ 0.50 확인, `predictions.csv`, `summary.json` 출력) |
| `smoke_test.py` | 파이프라인 단일 실행 규격(중립 폴더 경로명, `test` ROCF 전용 입력, 필수 JSON 키 7개, 사전 학습 가중치 반입 금지, `ground_truth.json` 삭제 및 오프라인 실행, `N=400` 환산 10분 이내·CPU 2분 이내 권장, `--null-effect` 거짓 양성 경고) 자동 점검 스크립트 |
| `rocf_vlm_synthetic_example.ipynb` | 예제 노트북: 천장효과·진단군 교란 확인 → 학습 전용 뇌지표 탐색 → 정적 그림과 과정 패널 비교 → 5 Hz 과정 특징 및 반증 검증 → 블라인드 정답 비교 |
| `requirements.txt` | 필요한 Python 패키지 |

## 만들기

설치는 [시작 안내](../../START_HERE.md#4-설치와-첫-실행-30분-안팎)를 따릅니다. 이 폴더에서 아래 명령을 실행합니다.

```bash
python generate_synthetic_rocf.py                                       # data/ 에 기본 데이터(seed 42) 생성
python generate_synthetic_rocf.py --seed 7 --out data_seed7             # 다른 난수(seed 7) 데이터 생성
python generate_synthetic_rocf.py --null-effect --out data_null         # 과정 신호를 0으로 끈 영효과 데이터 생성
python generate_synthetic_rocf.py --distribution-shift --out data_shift # 배경·펜 잡음·결측치 분포 변화 데이터 생성
```

같은 `--seed`를 주면 항상 같은 데이터가 만들어집니다. `--null-effect` 옵션을 켜면 진단군·뇌지표·정적 도형 분포는 유지하면서 시공간 그리기 과정에 심어 둔 아밀로이드 특이 신호만 정확히 0으로 제거합니다. `--distribution-shift` 옵션을 켜면 PNG 배경 밝기가 `255`에서 `246`으로 낮아지고, 펜 좌표(`x_mm`, `y_mm`)에 `0.35 mm` 정규 잡음이 더해지며, 대기(`pen_down=0`) 행의 `6%`가 무작위로 탈락(행 수가 2개 이상인 경우)하고, `train_privileged.csv`의 4개 열(`hippocampal_md`, `precuneus_md`, `edu_years`, `copy_score_36`)에 약 `5%` 결측치(`NaN`)가 추가됩니다. 따라서 이미지 이진화에는 95백분위 배경 추정 적응 임계값을, `train_privileged.csv` 처리에는 `SimpleImputer(strategy="median")` 같은 결측치 처리를 적용해야 합니다.

| 만들어지는 파일 | 내용 |
| --- | --- |
| `data/participants.csv` | 참가자 `N=400`명의 분할 표(`subject_id, split` 두 열만 포함; `train` 280명 / `test` 120명, `diagnosis + "_" + amyloid_pos` 층화 분할). 평가 시점에 ROCF 그림·과정 외의 외부 변수를 읽을 수 없도록 구조적으로 분리합니다. |
| `data/train_privileged.csv` | **`train` 280명 전용** 16개 열 특권 정보 표(`subject_id`, `amyloid_pos`, `amyloid_centiloid`, `diagnosis`, `age`, `sex`, `edu_years`, `apoe4_count`, `copy_score_36`, `recall_score_36`, `hippocampal_md`, `hippocampal_vol`, `entorhinal_thickness`, `precuneus_md`, `parietal_thickness`, `wmh_vol_log`). `test` 행은 전혀 포함되지 않습니다. |
| `data/images/` | 참가자별 파일 `{subject_id}_copy.png`, `{subject_id}_recall.png`(예: `sub-0001_copy.png`, `sub-0001_recall.png`, `672×480` 8-bit grayscale). 그리기 순서나 펜을 들고 멈춘 시간 정보는 들어 있지 않습니다. |
| `data/strokes/` | 참가자별 파일 `{subject_id}_copy.csv`, `{subject_id}_recall.csv`(예: `sub-0001_copy.csv`, `sub-0001_recall.csv`). 5 Hz(`0.20`초 간격) 디지털 펜 시계열(`t_sec, x_mm, y_mm, pen_down`, `210.0×150.0 mm` 캔버스 좌표)이며, `pen_down=1`은 펜을 종이에 대고 긋는 구간, `pen_down=0`은 펜을 들고 이동하거나 멈춰 대기하는 구간입니다. |
| `data/ground_truth.json` | 블라인드 `test` 120명의 실제 아밀로이드 정답(`test_labels`), 진단군(`test_diagnoses`), 참고 36점 점수, 생성 모형 파라미터. 파이프라인 코드에서는 절대 읽지 않으며 과제 4의 최종 검증에서만 사용합니다. |

| 데이터 구분 | `train` (학습용, 280명) | `test` (평가용, 120명) |
| --- | --- | --- |
| `images/` + `strokes/` | 제공 (학습 입력 가능) | 제공 (추론 입력) |
| `train_privileged.csv` (16개 열) | **280행 제공** | **0행 (미포함)** |
| `ground_truth.json` (`test_labels`, `test_diagnoses`) | 미포함 | 블라인드 정답 (**파이프라인에서 읽기 금지**) |

## 어떻게 만들어지나요?

1. **진단군 교란 (`CU + MCI`)과 아밀로이드 부하**: 참가자 `N=400`명의 인구통계·`APOE4`·진단군(`CU` 또는 `MCI`)·이진 `amyloid_pos`와 연속형 `amyloid_centiloid`를 생성합니다. `MCI` 집단은 도형 손상도 흔하고 아밀로이드 양성률도 약 60–68%로 높게 설정되어 강한 교란 변수가 됩니다.
2. **정적 36점 점수와 `CU` 천장효과 (진단군 조건부 독립)**: 18개 ROCF 구성 요소의 누락·위치 오차·왜곡과 펜 이동 속력(`v_draw`)은 `diagnosis`(`MCI` 여부)와 아밀로이드 무관 잡음에만 의존하도록 생성합니다. 따라서 `CU` 참가자는 대부분 33–36점 만점 근처에 분포하며, 완성된 정적 그림만으로는 `CU` 내부와 `MCI` 내부 모두에서 AUROC가 `0.50`에 머뭅니다.
3. **5 Hz 시공간 그리기 과정 신호와 학습 전용 뇌지표(특권 정보 활용 학습, Learning Using Privileged Information)**: 아밀로이드 병리가 높을수록(연속 `amyloid_centiloid` 포함) 완성된 도형이 정상처럼 보이더라도, (a) 상위 구조 요소와 하위 세부 요소가 만나는 경계에서 계획 멈춤(`pen_down=0`) 시간이 길어지고, (b) `Copy` 대비 `Delayed Recall`에서 큰 골격을 먼저 세우는 위계적 그리기 순서 재구성 지표가 낮아집니다. 학습 전용 뇌지표 중 미세구조 확산 지표(`hippocampal_md`, `precuneus_md`)는 조기 아밀로이드·과정 병리를 함께 반영하고, 거시 위축 지표(`hippocampal_vol`, `entorhinal_thickness`)는 `MCI` 위축과 연령을, `wmh_vol_log`는 혈관성 운동 감속 잡음을 반영합니다. 획 순서를 섞는 `shuffle_trial_strokes`를 적용하면 정적 픽셀은 그대로지만 시간·순서 구조가 깨지고, `--null-effect` 모드에서는 과정 신호가 0이 되어 진단군 내 ROCF 예측력이 정확히 `0.50`이 됩니다.

## 한계

- 18개 ROCF 구성 요소를 규칙 기반 선분·기하 도형으로 합성했으므로, 실제 사람의 자유로운 손글씨·곡선 비대칭·종이 기울임·필압 및 펜 기울기 변화는 반영되어 있지 않습니다.
- `train_privileged.csv`의 뇌영상 지표(`hippocampal_md`, `precuneus_md`, `hippocampal_vol` 등)는 실제 MRI/PET 측정값이 아니라 특권 정보 활용 학습(Learning Using Privileged Information)·교란 제거 학습을 연습하기 위해 문헌 경향만 단순화해 생성한 합성 표입니다.
- 실제 신경심리검사에서는 숙련된 평가자가 모호한 선 연결이나 비율 왜곡을 채점 지침에 따라 판정하지만, 합성 데이터는 생성 모형의 요소 점수를 직접 기록합니다.
- 이 합성 데이터에서 높은 점수를 낸 특징이나 임계값을 실제 임상 집단에 그대로 적용해서는 안 됩니다. 기본 파이프라인과 노트북은 Python 3.11·3.12 환경(일반 노트북 CPU 및 무료 Google Colab T4 GPU 호환)을 기준으로 작성되었습니다.

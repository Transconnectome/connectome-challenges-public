# ROCF VLM · 복합도형 그리기 과정으로 아밀로이드 베타 예측하기

![ROCF VLM 챌린지 요약](../assets/challenge_rocf_vlm.png)

- 난이도: **L2** (학습 전용 뇌지표 특권 정보 활용 · 평가 시 ROCF 그림·과정 전용 입력 · 교란 통제 및 반증 실험 설계)
- 데이터: 합성 ROCF 디지털 펜 기록(정적 흑백 그림 + 5 Hz 펜 궤적, `N=400`, 인지정상 `CU` + 경도인지장애 `MCI`) 및 학습(`train`) 전용 합성 뇌·임상 특권 정보(`train_privileged.csv`)
- 필요한 것: 일반 노트북 (CPU만으로 실행·검증 가능, 권장 2분 이내; 딥러닝 실험 시 무료 Google Colab T4 16 GB 호환, 비용 0원)
- 예상 시간: 기본 과제·반증 검증 4–6시간, 자율 모델링·확장 실험 별도

## 무엇을 해 보나요?

**ROCF**(Rey–Osterrieth Complex Figure, 레이-오스테리에트 복합도형)는 큰 직사각형 뼈대와 대각선·십자선·세부 기하 요소 등 **18개 구성 요소**(요소당 0–2점, 만점 **36점**)로 이루어진 시공간·기억 신경심리검사입니다. 참가자는 먼저 도형을 보며 그대로 따라 그리고(`Copy`), 일정 시간이 지난 뒤 도형을 보지 않고 기억에 의존해 다시 그립니다(`Delayed Recall`).

이 챌린지의 목표는 **평가(`test`) 시점에는 오직 ROCF 완성 그림(`images/*.png`)과 5 Hz 그리는 과정(`strokes/*.csv`)만 입력으로 받아** 뇌 **아밀로이드 베타 양성 여부**(`amyloid_pos`)를 예측하는 것입니다. 대신 **학습(`train`) 시점에는** 합성 뇌영상 지표(`hippocampal_md`, `precuneus_md`, `hippocampal_vol`, `entorhinal_thickness`, `parietal_thickness`, `wmh_vol_log`, 연속형 `amyloid_centiloid`)와 임상·인구통계·36점 채점 정보를 담은 **특권 정보**(`train_privileged.csv`)를 모두 활용해 교란 제거·다중과제 학습·표현 정렬·지식 증류(특권 정보 활용 학습, Learning Using Privileged Information)를 수행할 수 있습니다.

인지정상(`CU`)과 경도인지장애(`MCI`)가 섞인 표본(`N=400`)에서 이 문제를 풀 때는 실제 연구자들이 자주 빠지는 **두 가지 고전적인 함정**이 숨어 있습니다.

1. **진단군 교란 함정 (`CU + MCI` Confounding)과 `CU` 천장효과**: `MCI` 집단은 도형 요소의 누락·왜곡이 크고 아밀로이드 양성률도 높아서(약 60–68%), 완성된 정적 그림만 모델에 넣어도 전체(`CU + MCI`) AUROC가 `0.64–0.70` 안팎으로 그럴듯하게 나옵니다. 그러나 정적 그림은 진단군(`CU` vs `MCI`)이 주어졌을 때 아밀로이드와 독립이며, `CU` 참가자는 대부분 33–36점 만점 근처에 몰립니다(**천장효과**). 따라서 진단군 내부만 따로 떼어 평가하는 **진단군 내 층화 AUROC**(`CU` 단독·`MCI` 단독 및 가중평균 `within_dx_auroc`)를 계산하면 정적 그림 모델의 성능은 **`0.50`**(무작위)에 머뭅니다.
2. **시공간 그리기 과정 신호, 학습 전용 뇌지표 활용, 반증 대조군**: 완성된 정적 흑백 그림에는 남지 않지만, 초당 5번(5 Hz) 기록된 펜 궤적(`t_sec, x_mm, y_mm, pen_down`)의 **구조 경계 계획 멈춤**(`pen_down=0` 대기 구간)과 **따라 그리기(`Copy`) 대비 지연 회상(`Delayed Recall`)의 위계적 순서 재구성**에는 진단군 내부에서도 아밀로이드 병리(및 조기 미세구조 뇌지표인 `hippocampal_md`·`precuneus_md`·`amyloid_centiloid`)와 연결된 참 신호가 들어 있습니다. 이 신호는 시계열 특징 공학·경량 신경망·특권 정보 활용 학습(Learning Using Privileged Information) 뇌지표 정렬로 모델링하거나, 시간 흐름을 색상·단계별 패널(`render_process_panel`)로 시각화해 시각-언어 모델(VLM)로 탐색·설계할 수 있습니다. 단, 진짜 **과정** 신호를 학습했다면 (a) 테스트 획 시각·순서를 무작위로 섞는 **셔플 대조군**에서는 정적 그림이 100% 같더라도 진단군 내 AUROC가 **`0.50` 근처로 떨어져야** 하고, (b) 과정 효과를 0으로 끈 **`--null-effect` 영효과 데이터**에서는 전체 AUROC에 속지 않고 거짓 양성(`claim_true_amyloid_signal: true`)을 주장하지 않아야 합니다.

> **중요 — OpenRouter VLM 활용과 오프라인 실행의 경계**
>
> 외부 VLM API(예: OpenRouter)는 개발 과정에서 **① 시간-색상·단계별 패널 시각 탐색과 오류 분석(소수 표본)**, **② 특징·아키텍처·프롬프트 가설 수립**, **③ 확장 질문 실험 및 보고서 분석**에 자유롭게 활용할 수 있습니다.
> 그러나 **최종 제출하는 `run_pipeline.py` 본체는 평가 환경에서 외부 네트워크나 API 키 없이(`OPENROUTER_API_KEY=""`) 로컬 CPU/GPU 단독으로 처음부터 끝까지 실행**되어야 합니다(`smoke_test.py`가 오프라인 상태로 이를 직접 점검합니다).

## 시작하기

처음이라면 [시작 안내](../START_HERE.md)의 가상환경·패키지 설치(`pip install -r challenges/rocf_vlm/requirements.txt`) 단계를 먼저 따라 해 주세요. 설치가 끝났다면 저장소 맨 위 폴더에서 아래를 차례로 실행합니다(이미 `challenges/rocf_vlm` 폴더에 있다면 첫 줄은 건너뜁니다).

```bash
cd challenges/rocf_vlm
python generate_synthetic_rocf.py
python starter_pipeline.py --data-dir data --out-dir outputs --device cpu
python smoke_test.py --pipeline starter_pipeline.py --device cpu
jupyter notebook rocf_vlm_synthetic_example.ipynb
```

실행하면 `data/` 폴더에 아래 다섯 가지가 만들어집니다.

- `participants.csv`: 전체 `N=400`명의 분할 표(`subject_id, split` 두 열만 포함; `train` 280명 / `test` 120명, `diagnosis + "_" + amyloid_pos` 층화 분할).
- `train_privileged.csv`: **`train` 280명만 들어 있는 학습 전용 특권 정보 표 (16개 열)**: `subject_id`, `amyloid_pos`, `amyloid_centiloid`, `diagnosis`, `age`, `sex`, `edu_years`, `apoe4_count`, `copy_score_36`, `recall_score_36`, `hippocampal_md`, `hippocampal_vol`, `entorhinal_thickness`, `precuneus_md`, `parietal_thickness`, `wmh_vol_log`. **`test` 참가자 행은 전혀 들어 있지 않습니다.**
- `images/`: `train`·`test` 전원의 모사(`{subject_id}_copy.png`)·지연회상(`{subject_id}_recall.png`) 완성 정적 흑백 이미지(예: `sub-0001_copy.png`, `sub-0001_recall.png`, 크기 `672×480` 8-bit grayscale).
- `strokes/`: `train`·`test` 전원의 5 Hz(0.20초 간격) 펜 시계열 파일(`{subject_id}_copy.csv`, `{subject_id}_recall.csv`, 예: `sub-0001_copy.csv`, `sub-0001_recall.csv`). 열은 `t_sec, x_mm, y_mm, pen_down`이며 좌표는 `210.0×150.0 mm` 캔버스 기준, `pen_down=1`은 펜을 종이에 대고 긋는 구간, `pen_down=0`은 펜을 들고 이동하거나 멈춰 대기하는 구간입니다.
- `ground_truth.json`: 과제 4의 최종 확인을 위한 블라인드 `test` 120명의 정답(`test_labels`, `test_diagnoses`, 참고 36점 점수). **과제 1–3을 마칠 때까지 열어 보지 말고, 파이프라인 코드에서도 절대 읽지 마세요.**

### `train` (280명) vs `test` (120명) 데이터 구성 비교

| 항목 | `train` (학습용, 280명) | `test` (평가용, 120명) |
| --- | --- | --- |
| `participants.csv` (`subject_id`, `split`) | 포함 (`split == "train"`) | 포함 (`split == "test"`) |
| `images/{subject_id}_{copy,recall}.png` (`672×480` 흑백) | 제공 (입력 가능) | 제공 (추론 입력) |
| `strokes/{subject_id}_{copy,recall}.csv` (5 Hz, `t_sec, x_mm, y_mm, pen_down`) | 제공 (입력 가능) | 제공 (추론 입력) |
| `train_privileged.csv` (16개 열: 뇌지표·임상·36점·정답 라벨) | **280행 모두 제공** (교란 제거·보조 타깃·표현 정렬 등에 활용) | **0행 (전혀 포함되지 않음)** |
| `ground_truth.json` (`test_labels`, `test_diagnoses`) | 미포함 | 블라인드 정답 검증용 (**모델 코드에서 읽기 금지**) |

### 분포 변화(`--distribution-shift`) 및 결측치 대비

평가 검증에서는 기본 합성 코호트 외에도 배경 밝기·펜 잡음·결측치가 바뀐 `--distribution-shift` 데이터로 파이프라인의 강건성을 점검할 수 있습니다.

- `python generate_synthetic_rocf.py --distribution-shift --out data_shift`를 실행하면 (1) PNG 배경 밝기가 `255`에서 `246`으로 어두워지고, (2) 펜 좌표(`x_mm`, `y_mm`)에 표준편차 `0.35 mm` 정규 잡음이 추가되며, (3) 대기(`pen_down=0`) 행의 `6%`가 무작위로 탈락(행 수가 2개 이상인 경우)하고, (4) `train_privileged.csv`의 4개 열(`hippocampal_md`, `precuneus_md`, `edu_years`, `copy_score_36`)에 약 `5%`의 결측치(`NaN`)가 들어갑니다.
- 따라서 파이프라인(`run_pipeline.py`)은 절대 임계값(예: `arr < 200`) 대신 배경 적응 임계값(예: `95`백분위 배경 추정치 `bg - 45.0`)을 쓰고, `train_privileged.csv` 특권 변수를 쓸 때 `SimpleImputer(strategy="median")` 같은 결측치 처리를 반드시 포함해야 합니다.

> **스타터 파이프라인(`starter_pipeline.py`) 안내**
>
> 공개 제공되는 `starter_pipeline.py`는 의도적으로 **정적 흑백 그림만 사용하는 최소 기준선**만 구현해 두었습니다(실행 규격 확인용이며 `smoke_test.py`를 통과합니다). 따라서 그대로 실행하면 전체 AUROC는 `0.64–0.67`이지만 진단군 내 AUROC는 `0.45–0.49`(`claim_true_amyloid_signal: false`)에 머뭅니다. 5 Hz 그리는 과정 모델링, 학습 전용 뇌지표(`hippocampal_md` 등)를 활용한 특권 정보 활용 학습(Learning Using Privileged Information)·교란 제거, VLM 기반 시각 탐색과 진정한 신호 검출은 직접 `run_pipeline.py`에 설계해야 합니다.

파일 구조와 데이터 생성 원리는 [합성 데이터 안내](rocf_vlm/README.md)에 있습니다.

## 과제

1. **데이터 구조·천장효과·교란 확인하기**: `train_privileged.csv`에서 `CU`와 `MCI`의 36점 점수 분포, 아밀로이드 양성률, `hippocampal_md`·`precuneus_md`·`hippocampal_vol` 분포를 비교합니다. 정적 그림만 쓰는 `starter_pipeline.py`가 왜 전체(`CU + MCI`)에서는 AUROC `0.64–0.67`이 나오지만 진단군 내 층화 AUROC(`CU` 단독·`MCI` 단독)에서는 `0.50` 근처로 실패하는지 그림과 수치로 확인합니다.
2. **ROCF 그림·과정 모델 + 학습 전용 뇌지표(특권 정보 활용 학습, Learning Using Privileged Information) 설계하기**: 평가(`test`) 시점에는 오직 `images/*.png`와 `strokes/*.csv`만 입력으로 쓰되, 학습(`train`) 시점에는 `train_privileged.csv`의 뇌지표(`hippocampal_md`, `precuneus_md`, `amyloid_centiloid` 등)와 진단군 정보를 자유롭게 활용해 교란 제거·다중과제·표현 정렬·지식 증류 파이프라인(`run_pipeline.py`)을 구현합니다. 시간-색상·단계별 패널(`render_process_panel`)을 그려 VLM으로 패턴을 탐색하는 것도 좋습니다(단, 최종 `run_pipeline.py` 기본 실행은 오프라인 단독 실행 필수).
3. **반증 대조군 2종 검증하기**: (a) 검증 폴드의 획 순서·시각을 무작위로 섞는 셔플 대조군(`shuffle_trial_strokes`)에서 진단군 내 AUROC(`cv_shuffle_within_dx_auroc`)가 `0.50` 근처로 떨어지는지, (b) 영효과 데이터(`python generate_synthetic_rocf.py --null-effect --out data_null`)에서 거짓 양성(`claim_true_amyloid_signal: true`)을 내뱉지 않고 `false`를 출력하는지 확인합니다.

```bash
cd challenges/rocf_vlm   # 이미 이 폴더라면 건너뜁니다
python generate_synthetic_rocf.py --null-effect --out data_null
python run_pipeline.py --data-dir data_null --out-dir outputs_null --device cpu
```

4. **정답 파일과 블라인드 테스트 비교하기**: 모델 선택과 반증 검증을 마친 뒤 `data/ground_truth.json`의 `test_labels`와 `test_diagnoses`로 블라인드 `test`(`120`명) 예측(`predictions.csv`)의 전체 AUROC, `CU` 단독 AUROC, `MCI` 단독 AUROC, 가중 진단군 내 AUROC(`within_dx_auroc`)를 확인하고 교차검증 결과와 비교합니다.
5. **단일 실행 규격(`smoke_test.py`)과 다른 seed(`seed 7`)로 반복하기**: `--seed 7 --out data_seed7`에서 같은 분석과 모델링의 재현성을 확인하고, `python smoke_test.py --pipeline run_pipeline.py --device cpu`(및 CUDA 코드 포함 시 무료 Google Colab T4 런타임에서 `--device cuda`)를 통과합니다.

```bash
cd challenges/rocf_vlm   # 이미 이 폴더라면 건너뜁니다
python generate_synthetic_rocf.py --seed 7 --out data_seed7
python run_pipeline.py --data-dir data_seed7 --out-dir outputs_seed7 --device cpu
python smoke_test.py --pipeline run_pipeline.py --device cpu
```

## 파이프라인 실행 규격과 공정한 GPU/CPU 기준

자율 설계한 파이프라인(`run_pipeline.py`)은 누가 실행하더라도 아래 단일 명령 규격으로 오프라인 동작해야 하며, `smoke_test.py`가 중립적인 임시 폴더 경로에서 이를 자동으로 검사합니다.

```bash
python run_pipeline.py --data-dir data --out-dir outputs --device cpu
```

- **입력 인자 규격**:
  - `--data-dir`: `participants.csv`(`subject_id,split`), `train_privileged.csv`(`train` 전용 뇌·임상 특권 정보), `images/`, `strokes/`가 들어 있는 데이터 폴더 경로. 파이프라인 안에서 `ground_truth.json`을 열거나 폴더 경로 이름으로 조건을 추측하면 규격 위반입니다.
  - `--out-dir`: 예측 표와 요약 지표를 저장할 출력 폴더 경로.
  - `--device`: `cpu` 또는 `cuda`. 기본 검증은 `--device cpu`로 동작해야 하며, 신경망·비전 백본을 쓸 때는 `--device cuda`를 함께 지원할 수 있습니다.
- **출력 파일 1 — `predictions.csv`**:
  - `participants.csv`에서 `split == "test"`인 블라인드 참가자 전원에 대해 `subject_id,y_prob` 두 열을 저장합니다(`0.0 <= y_prob <= 1.0`, 결측치·`NaN` 금지).
- **출력 파일 2 — `summary.json`**:
  - `train`(`280`명) 내부 층화 교차검증(검증 폴드는 `test`와 동일하게 ROCF 그림·과정만 입력!)과 반증 실험 요약을 아래 7개 필수 키로 기록합니다.
  - `cv_auroc_pooled`: `CU + MCI` 전체 교차검증 AUROC
  - `cv_auroc_cu`: 인지정상(`CU`) 내부 교차검증 AUROC
  - `cv_auroc_mci`: 경도인지장애(`MCI`) 내부 교차검증 AUROC
  - `cv_within_dx_auroc`: `CU`·`MCI` 표본 수 가중 평균 진단군 내 AUROC (핵심 지표)
  - `cv_within_dx_auroc_ci95`: `cv_within_dx_auroc`의 95% 신뢰구간 (`[하한, 상한]`)
  - `cv_shuffle_within_dx_auroc`: 검증 폴드 획 순서·시각 셔플 대조군에서의 진단군 내 AUROC
  - `claim_true_amyloid_signal`: 진단군 내 AUROC 신뢰구간과 셔플 반증 통과 여부를 종합한 최종 판정(`true` 또는 `false`)
- **사전 학습 가중치 반입 금지**:
  - 외부에서 미리 학습해 저장한 가중치나 직렬화 파일(`.pt`, `.pth`, `.safetensors`, `.ckpt`, `.onnx`, `.bin`, `.pkl`, `.joblib`)을 저장소에 커밋하거나 불러오는 것은 금지합니다. 모든 모델은 실행 시점에 주어진 `--data-dir`의 `train` 데이터로만 학습해야 합니다.
- **공정한 자원 상한 (VRAM 16 GB / 상한 10분 / 노트북 CPU 2분 내 권장)**:
  - 고가 연산 장비 유무로 유불리가 갈리지 않도록, 파이프라인 1회 실행(`N=400`)은 **무료 Google Colab T4 GPU(VRAM 16 GB 이하)** 또는 **노트북 CPU**에서 **최대 10분(600초) 이내**에 끝나야 합니다. 특히 실시간 대면 검증이나 빠른 반복 실험을 위해 **노트북 CPU 기준 2분(120초) 이내**(축소 표본 `N=120`에서는 30초 이내) 완료를 권장합니다.

무료 Google Colab(런타임 유형: Python 3, 하드웨어 가속기: **T4 GPU**)에서 환경을 바로 검증할 때는 공개 저장소 첫 페이지의 초록색 **`Code`**(또는 `[<> Code]`) 버튼을 눌러 **HTTPS 저장소 주소**를 복사한 뒤, 새 노트북 셀에 아래 `%%bash` 블록을 붙여 넣고 `REPO_URL` 자리에 복사한 주소를 넣어 실행할 수 있습니다(공개 저장소에는 기준선인 `starter_pipeline.py`가 들어 있어 복사 직후 바로 실행됩니다).

```bash
%%bash
REPO_URL="여기에_복사한_공개_저장소_HTTPS_주소를_붙여_넣으세요"
git clone "$REPO_URL" repo && cd repo/challenges/rocf_vlm
pip install -q -r requirements.txt
python generate_synthetic_rocf.py
python starter_pipeline.py --data-dir data --out-dir outputs --device cuda
python smoke_test.py --pipeline starter_pipeline.py --device cuda
```

본인의 제출용 `run_pipeline.py`를 검증할 때는 먼저 `repo/challenges/rocf_vlm/run_pipeline.py` 위치에 본인의 `run_pipeline.py`를 직접 업로드하거나(처음 시작할 때는 `cp starter_pipeline.py run_pipeline.py`로 스타터 파일을 복사해 시작 가능), 아래처럼 `run_pipeline.py`를 준비한 뒤 다음 셀에서 실행해 주세요.

```bash
%%bash
cd repo/challenges/rocf_vlm
# 본인의 run_pipeline.py를 업로드했다면 아래 cp 줄은 건너뛰세요
cp starter_pipeline.py run_pipeline.py
python run_pipeline.py --data-dir data --out-dir outputs --device cuda
python smoke_test.py --pipeline run_pipeline.py --device cuda
```

> Colab에서는 줄마다 `!`를 따로 쓰면 첫 줄의 `REPO_URL`이나 `cd` 작업 경로가 다음 줄로 이어지지 않으므로 위처럼 `%%bash` 셀 한 블록으로 실행하는 편이 안전합니다. 본인의 비공개(Private) 저장소를 Colab에서 내려받을 때는 Colab 비밀번호 노출을 막기 위해 `git clone` 대신 작업 파일을 직접 업로드하거나, 일회성 Personal Access Token을 환경변수로만 잠시 입력한 뒤 출력과 토큰을 남기지 마세요.

## 스스로 확인하기

기본 정적 기준선(`starter_pipeline.py`)과 본인의 `run_pipeline.py`를 실행한 뒤 펼쳐 보세요.

<details><summary>확인용 숫자</summary>

아래 표는 **정적 흑백 그림만 사용하는 공개 스타터**(`starter_pipeline.py`, `--device cpu`)의 네 가지 조건(`N=400`, `train` 280 / `test` 120) 실측 결과입니다. 운영체제·패키지 버전에 따라 소수점 셋째 자리가 조금 다를 수 있습니다.

| 데이터 조건 (정적 스타터 `starter_pipeline.py`) | CV 전체 `cv_auroc_pooled` | CV `CU` `cv_auroc_cu` | CV `MCI` `cv_auroc_mci` | CV 진단군 내 `cv_within_dx_auroc` [95% CI] | CV 셔플 `cv_shuffle_within_dx_auroc` | 판정 `claim_true_amyloid_signal` | 블라인드 `test` 전체 / 진단군 내 AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 기본 (`seed 42`, `data/`) | 0.667 | 0.480 | 0.500 | 0.487 [0.406, 0.563] | 0.487 | `false` | 0.649 / 0.415 |
| 반복 (`seed 7`, `data_seed7/`) | 0.642 | 0.459 | 0.446 | 0.454 [0.381, 0.528] | 0.454 | `false` | 0.670 / 0.517 |
| 영효과 (`seed 42 --null-effect`) | 0.667 | 0.480 | 0.500 | 0.487 [0.406, 0.563] | 0.487 | `false` | 0.649 / 0.415 |
| 영효과 (`seed 7 --null-effect`) | 0.642 | 0.459 | 0.446 | 0.454 [0.381, 0.528] | 0.454 | `false` | 0.670 / 0.517 |

**목표 성능 감각(직접 설계한 과정 + 특권 정보 활용 학습(Learning Using Privileged Information) 파이프라인 `run_pipeline.py` 기준)**:

- **정상 데이터 (`seed 42`, `seed 7`)**: 5 Hz 그리는 과정 특징(구조 경계 계획 멈춤·모사 대비 지연회상 순서 재구성)과 학습 전용 뇌지표(`hippocampal_md`, `precuneus_md`, `amyloid_centiloid` 등)의 교란 제거·정렬을 결합하면, 평가 시 ROCF 입력만으로도 **진단군 내 AUROC `0.80–0.88`**(`CU` 단독 `0.80+`, 전체 `0.85+`)에 도달할 수 있으며 `claim_true_amyloid_signal: true`가 됩니다.
- **획 셔플 대조군**: 테스트 획 순서·시각을 무작위로 섞으면 정적 그림은 100% 동일하지만 진단군 내 AUROC는 **`0.50` 근처**로 크게 떨어져야 합니다(하락폭 `>= 0.20`).
- **영효과 데이터 (`--null-effect`)**: 전체 AUROC는 여전히 `0.60–0.67` 안팎으로 보일 수 있으나 진단군 내 AUROC는 **`0.50` 근처**이며, 신뢰구간과 반증 기준을 통해 `claim_true_amyloid_signal: false`를 올바르게 판정해야 합니다.

</details>

## 정리해 둘 것

- 진단군 교란·천장효과, 5 Hz 시공간 과정·VLM 패널, 학습 전용 뇌지표(특권 정보 활용 학습) 활용 효과 및 반증 대조군 결과를 비교한 핵심 그림 1–2장
- `smoke_test.py`를 통과하는 오프라인 파이프라인(`run_pipeline.py`)과 분석 노트북
- 기본(`seed 42`), 반복(`seed 7`), 영효과(`--null-effect`) 세 조건의 `summary.json` 지표와 블라인드 `test`(`120`명) 진단군 내 AUROC 비교표
- 반 쪽 이내의 설명: (1) 왜 `CU + MCI` 혼합 표본에서 정적 그림의 전체 AUROC가 착시를 일으키는지, (2) 학습 전용 뇌지표(`hippocampal_md` 등)가 평가 시 ROCF만 쓰는 모델에 어떻게 도움이 되었는지(또는 어떤 조건에서 도움이 되지 않았는지), (3) 셔플·영효과 대조군이 어떤 오류를 걸러 주는지 적어 주세요.

자유 학습자가 오픈 챌린지 결과를 공유하는 방법과 2026 동계 인턴십 2단계 지원자의 비공개 저장소 제출 구분은 [시작 안내](../START_HERE.md#5-결과-정리와-공유)에 있습니다.

## 2026 동계 인턴십 2단계 제출·일정 안내 (지원자 선생님용)

오픈 챌린지 자율 학습자께서는 위 과제만 자유롭게 진행하시면 됩니다. **2026 동계 인턴십 2단계 지원자 선생님**께서는 아래 일정·비공개 저장소 절차·API 키 주의사항을 꼭 확인해 주세요.

### 비공개(Private) GitHub 저장소 및 `OPENROUTER_API_KEY` 준비

1. **공개 Issues / Public 저장소 절대 금지**: 인턴십 2단계 지원 과제 코드, 풀이 전략, 비공개 저장소 주소, 지원번호는 이 공개 저장소의 **Issues**나 Public 저장소에 절대 올리지 마세요.
2. **Private GitHub 저장소 생성 및 채점 계정 초대**: GitHub에서 **`+` → New repository**를 눌러 반드시 **Private**을 선택한 뒤 저장소를 만드세요. 저장소 생성 후 **Settings → Collaborators → Add people**에서 `11/25(수)` 안내 메일에 적힌 채점용 GitHub 계정을 초대해 주세요.
3. **데이터·출력·API 키 제외(`.gitignore`)**: `.gitignore`에 `data/`, `data_*/`, `.venv/`, `.env`, `outputs*/`, `submissions/`가 포함되어 있는지 확인한 뒤 코드(`run_pipeline.py`, 분석 노트북, `README` 등)와 사전 학습 가중치가 아닌 소스만 `git commit` · `git push` 하세요. 자세한 Git 명령 순서는 [시작 안내 5절 (B)](../START_HERE.md#5-결과-정리와-공유)에 있습니다.
4. **`OPENROUTER_API_KEY` 안전 사용**: 안내 메일로 발급받은 OpenRouter API 키는 탐색 단계에서 터미널 환경변수(`export OPENROUTER_API_KEY="발급받은_키"`)로만 전달하고, `.py`/`.ipynb` 코드 안에 직접 적거나 Git에 올리지 마세요. 최종 제출용 `run_pipeline.py`는 `OPENROUTER_API_KEY=""` 오프라인 상태에서 끝까지 실행되어야 합니다.

### 2026 동계 인턴십 전체 일정 및 단계별 제출물

| 날짜 (요일) | 단계 | 선생님께서 확인·수행하실 일 및 제출물 |
| --- | --- | --- |
| `2026/11/25(수)` 오전 | 2단계 과제 안내 발송 | 통합 2단계 과제(**ROCF VLM · `CU + MCI`**), 개인별 `--seed`, 탐색용 OpenRouter API 키 안내 수신 |
| `2026/12/1(화) 23:59` | 2단계 1차 중간 제출 | 메일 제목 `[2026동계인턴챌린지-기준] 지원번호`로 **사전 확정 확장 질문, 1차 반증 기준, 기준선 커밋 해시** 제출 |
| `2026/12/7(월) 23:59` | 2단계 최종 제출 마감 | 메일 제목 `[2026동계인턴챌린지] 지원번호_ROCF`로 3종 제출: **(1) 채점 계정 초대 완료된 Private GitHub 저장소 링크**, **(2) A4 2쪽 이내 보고서 PDF**, **(3) A4 1쪽 이내 AI 활용 기록** |
| `2026/12/8(화) ~ 12/10(목)` | 2단계 비공개 검증·채점 | 숨겨진 합성 홀드아웃 3종 검증 + 평가자 2인 이상 루브릭 채점 |
| `2026/12/11(금)` | 2단계 결과 발표 | 면접 대상자 안내 (최종 정원 3명의 약 2~3배수인 약 6~9명 내외; 기준 통과자가 2~3배수 이하일 경우 통과자 전원 면접 진행) |
| `2026/12/14(월) ~ 12/18(금)` | 3단계 대면 면접 | 서울대학교 관악캠퍼스 30분 대면 면접: **새 `--seed`로 `N=120` 즉석 재실행 + 코드 설명 + 1단계 학습·설명 과제 메모 없이 3분 구두 설명** |
| `2026/12/22(화)` | 최종 합격자 발표 | 최종 합격자 3명 안내 |
| `2026/12/23(수) ~ 12/31(목)` | 사전 준비 기간 | 개인 노트북 기반 온보딩 과제 1~3 및 Git·작업 환경 준비 |
| `2027/1/4(월) ~ 2/26(금)` | 8주 동계 인턴십 본 과정 | 본 연구 진행 (`1/28(목)` 또는 `1/29(금)` 1개월 중간 점검·발표, `2/8(월)~2/9(화)` 설 연휴 휴무, `2/25(목)` 최종 발표, `2/26(금)` 종료) |

> **참고 — `12/1(화) 23:59` 기준선 커밋 해시 확인 방법**: 내 Private 저장소에 첫 커밋을 올린 뒤 저장소 루트에서 `git rev-parse --short HEAD` 명령을 실행하면 나오는 7자리 커밋 ID를 복사하거나, GitHub 내 Private 저장소 페이지의 커밋 이력(`Commits`)에서 해당 커밋 오른쪽에 표시된 7자리 ID를 복사해 메일에 함께 적어 주세요.

## 선택 심화

- `train_privileged.csv`의 조기 미세구조 지표(`hippocampal_md`, `precuneus_md`, `amyloid_centiloid`)와 거시 위축 지표(`hippocampal_vol`, `entorhinal_thickness`) 중 어느 조합을 보조 타깃·표현 정렬·교란 잔차화에 썼을 때 `CU` 단독 블라인드 성능이 가장 크게 개선되는지 체계적으로 비교해 보세요.
- `render_process_panel`이 만드는 시간-색상 궤적 이미지나 단계별 진행 패널을 VLM(예: OpenRouter)으로 분석해 도출한 시각적 관찰을 로컬 특징이나 경량 신경망으로 옮겼을 때 어떤 보완 효과가 있는지 시험해 보세요.
- `--null-effect` 데이터를 서로 다른 seed(`1`–`10`)로 여러 번 만들어 내 파이프라인의 `claim_true_amyloid_signal` 판정 규칙이 거짓 양성을 얼마나 자주 내는지(실제 제1종 오류율) 점검해 보세요.

## 합성 데이터의 한계

합성 ROCF 데이터는 `CU`의 만점 근처 천장효과, `MCI`의 높은 아밀로이드 양성률과 도형 손상, 구조 경계에서의 계획 멈춤, `Copy` 대비 `Delayed Recall`의 순서 재구성, 그리고 조기 미세구조 뇌지표(`hippocampal_md`, `precuneus_md`)와 거시 위축(`hippocampal_vol`)의 분리라는 핵심 가설 구조만 단순화해 구현한 것입니다. 실제 사람의 필기에서 나타나는 미세한 필압·속도 곡선, 손떨림, 도형의 연속적인 기하 왜곡이나 실제 뇌영상 전처리 변동성은 포함하지 않으므로, 이 합성 데이터에서 높은 점수를 낸 특징이나 임계값이 실제 임상 기록에서도 그대로 성립한다고 가정하지 마세요. 자세한 생성 원리와 한계는 [합성 데이터 안내](rocf_vlm/README.md#한계)에 있습니다.

## 용어

- **ROCF(Rey–Osterrieth Complex Figure)**: 18개 기하 구성 요소(만점 36점)로 이루어진 복합도형 신경심리검사. 보고 따라 그리기(`Copy`, `copy_score_36`)와 기억에 의존해 다시 그리기(`Delayed Recall`, `recall_score_36`)로 시공간 구성 능력과 시각 기억을 함께 평가합니다.
- **특권 정보 활용 학습(Learning Using Privileged Information)**: 학습(`train`) 단계에서는 뇌영상 지표(`hippocampal_md` 등)·연속 아밀로이드 수치·진단 정보 같은 풍부한 추가 데이터를 모두 활용하되, 실제 추론·평가(`test`) 단계에서는 저비용·비침습 입력(ROCF 그림·그리는 과정)만으로 예측하도록 모델을 학습시키는 패러다임입니다.
- **CU(Cognitively Unimpaired, 인지정상)** / **MCI(Mild Cognitive Impairment, 경도인지장애)**: `CU`는 정상 인지 범위, `MCI`는 객관적 인지 저하가 시작된 단계로 아밀로이드 양성 비율과 도형 손상이 모두 높습니다.
- **아밀로이드 베타(`amyloid_pos`) / Centiloid(`amyloid_centiloid`)**: 알츠하이머병 병리의 핵심 표지 단백질(아밀로이드 베타)이 뇌에 축적되었는지 나타내는 이진 양성 여부(`amyloid_pos`, 0 또는 1)와 그 축적량을 표준화된 연속 척도로 나타낸 수치(`amyloid_centiloid`, 약 **25 CL** 이상을 양성 기준으로 둠)입니다.
- **해마 평균확산도(`hippocampal_md`) / 설전부 평균확산도(`precuneus_md`)**: 확산강조 MRI(dMRI)에서 뇌 조직 미세구조 손상을 반영하는 지표로, 해마 부피(`hippocampal_vol`)의 거시적 위축이 뚜렷해지기 전 조기 단계부터 상승할 수 있습니다.
- **해마 부피(`hippocampal_vol`) / 내후각 피질 두께(`entorhinal_thickness`) / 두정엽 피질 두께(`parietal_thickness`)**: 구조 MRI에서 측정하는 거시적 뇌 위축 지표입니다. `hippocampal_md`가 미세한 물 분자 확산 변화를 보는 조기 지표라면, `hippocampal_vol`과 피질 두께(`entorhinal_thickness`, `parietal_thickness`)는 뇌 조직 자체가 얇아지거나 줄어든 거시 위축(연령·`MCI`와 강하게 연관)을 반영합니다.
- **백질 고신호 부피(`wmh_vol_log`)**: MRI에서 뇌 백질의 미세혈관 손상(백질 변성) 부피에 로그를 취한 지표로, 아밀로이드 병리 외에 혈관성 운동 감속(그리기 속도 저하)과 연관될 수 있습니다.
- **APOE4 유전자형(`apoe4_count`) 및 인구통계(`age`, `sex`, `edu_years`)**: `apoe4_count`(0, 1, 2)는 알츠하이머병 유전적 위험 인자인 *APOE* ε4 대립유전자 보유 개수이며, `age`(연령), `sex`(성별), `edu_years`(교육 연수)는 참가자 기본 배경 변수입니다.
- **AUROC 및 95% 신뢰구간(`CI95`, `cv_within_dx_auroc_ci95`)**: AUROC는 모델이 양성(`Aβ+`)과 음성(`Aβ-`)을 가르는 판별력 지표로, `0.50`은 무작위 추측(동전 던지기), `1.00`은 완벽한 분리를 뜻합니다. `CI95`는 부트스트랩 반복 추정을 통해 얻은 95% 신뢰구간(`[하한, 상한]`)으로, 하한이 `0.55`를 넘는지처럼 우연 이상의 안정적 신호인지 판단할 때 씁니다.
- **천장효과(Ceiling effect)**: 과제가 특정 집단(`CU`)에게 쉬워서 점수가 만점(33–36점) 근처에 몰려, 정적 점수나 그림만으로는 집단 내부의 차이를 구분할 수 없는 현상입니다.
- **교란 변수(Confounding) 및 진단군 내 층화 AUROC(`within_dx_auroc`)**: `MCI` 진단군처럼 입력과 타깃 양쪽에 모두 영향을 주어 가짜 성능을 만드는 제3의 변수를 교란 변수라고 합니다. `within_dx_auroc`는 `CU` 내부 AUROC와 `MCI` 내부 AUROC를 표본 수로 가중 평균한 지표로, `CU` vs `MCI` 집단 차이(교란)만 외우는 모델은 정확히 `0.50`을 받습니다.
- **특권 정보 활용 주요 전략(교란 잔차화·다중과제 학습·표현 정렬·지식 증류)**:
  - **교란 잔차화(Confounder residualization)**: 학습 데이터의 `diagnosis`·`age` 등으로 설명되는 교란 성분을 분리·통제해 아밀로이드 특이 신호만 남기는 기법.
  - **다중과제 학습(Multi-task learning)**: `amyloid_pos` 분류와 함께 학습 시에만 뇌지표(`hippocampal_md`, `amyloid_centiloid` 등) 회귀를 동시에 예측해 더 풍부한 표현을 배우는 기법.
  - **표현 정렬(Representation alignment)**: ROCF 과정 인코더의 중간 벡터가 학습 전용 뇌지표 공간과 정렬되도록 제약을 걸어, 평가 시 ROCF만으로도 뇌지표 정보를 추론하게 하는 기법.
  - **지식 증류(Knowledge distillation)**: 특권 뇌지표까지 모두 본 교사(Teacher) 모델의 예측·표현을 ROCF만 입력받는 학생(Student) 모델에 전달하는 기법.
- **반증 대조군(Falsification control)**: 모델이 진짜 과정 기전을 학습했는지 시험하기 위해 획 순서를 무작위로 섞거나(`shuffle_trial_strokes`) 과정 신호를 0으로 끈(`--null-effect`) 대조 실험입니다.
- **VLM(Vision-Language Model, 시각-언어 모델)**: 이미지와 텍스트를 함께 처리해 도형의 공간 배치나 단계별 그리기 진행 패널의 시각적 패턴을 해석·탐색하는 다중모달 모델입니다.

## 나와 연구의 fit 돌아보기

겉보기 전체 AUROC가 높게 나왔을 때 그대로 믿지 않고 진단군 내 층화 평가와 반증 대조군으로 파고드는 과정이 재미있었나요? 학습 전용 뇌지표(특권 정보 활용 학습)를 이용해 ROCF 전용 추론 모델을 끌어올리는 아이디어를 설계할 때 어떤 가설이 통했고 어떤 가설이 실패했나요? 결과를 더 단단히 믿으려면 어떤 대조 실험을 하나 더 추가하고 싶나요?

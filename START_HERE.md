# 시작 안내: 처음 오신 분께

이 저장소는 뇌 데이터를 직접 다뤄 보는 **연습 챌린지** 모음입니다. 정답을 맞히는 시험이 아닙니다. 데이터를 그려 보고, 이상한 점을 찾고, 그 이유를 근거와 함께 설명하는 경험이 목표입니다. 해 보면서 이런 연구가 나와 맞는지 가늠해 보세요.

## 1. 무엇을 얻나요?

- 실제 연구에서 쓰는 도구(Python, MNE, nilearn, scikit-learn 등)로 뇌파·fMRI·디지털 인지행동 데이터를 처음부터 끝까지 다뤄 봅니다.
- 연구자가 자주 빠지는 함정(눈 깜빡임, 머리 움직임, 진단군 교란과 천장효과)을 직접 찾아내고 반증 대조군으로 점검합니다.
- 먼저 **합성 데이터**로 연습합니다. L1 챌린지(EEG, fMRI)는 정답 파일로 분석을 스스로 확인할 수 있고, L2 챌린지(ROCF VLM)는 연구실이 미리 생성해 둔 블라인드 합성 데이터(`data.zip`, 평가 정답 비공개)에서 `train` 교차검증과 자체 대조 검증을 수행합니다.
- L1 챌린지(EEG, fMRI)에서는 선택 단계로 **실제 공개 데이터**에 같은 분석을 적용해, 합성 데이터와 무엇이 다른지 봅니다. 실제 데이터에는 정답이 없습니다.

## 2. 챌린지 고르기

| 챌린지 | 다루는 것 | 난이도 |
| --- | --- | --- |
| [EEG · 뇌파에서 눈 깜빡임 흔적 찾아보기](challenges/eeg.md) | 뇌파 그리기, 아티팩트 찾기 | L1 |
| [fMRI · 뇌 영역의 신호는 함께 움직일까?](challenges/fmri.md) | 영역 간 상관, 머리 움직임 보정 | L1 |
| [ROCF VLM · 복합도형 그리기 과정으로 아밀로이드 베타 예측하기](challenges/rocf_vlm.md) | 시공간 그리기 과정 모델링, 진단군 교란 통제, 신호 진위 자체 검증 | L2 |

예상 시간(추정): L1 챌린지는 합성 데이터 과제 2–3시간 + 실제 공개 데이터 단계(선택) 약 1시간, L2 챌린지(ROCF VLM)는 기본 과제·반증 검증 4–6시간(자율 모델링·확장 실험 별도).

난이도는 과제에 필요한 독립성을 나타냅니다. 학년이나 학위에 따른 참가 제한이 아닙니다.

- **L1**: 안내를 따라 실행하고, 본 것을 설명합니다.
- **L2**: 분석 방법을 스스로 고르고 비교합니다.
- **L3**: 새 질문을 세우고, 틀렸는지 판단할 기준까지 정해 검증합니다.

지금은 L1 입문 챌린지 두 개와 L2 심화 챌린지 한 개가 있습니다. 처음이라면 L1 두 개 중 어느 쪽부터 해도 되고, Python이 처음이라면 EEG가 설치가 가볍고 빠릅니다. 교란 변수 통제와 시계열·다중모달 반증 실험까지 직접 설계해 보고 싶다면 L2인 ROCF VLM을 골라 보세요.

> **2026 동계 인턴십 2단계 통합 과제 안내**
>
> **EEG·fMRI는 입문용 L1 연습 과제**이며, **2026 동계 인턴십 2단계 공식 평가 과제는 [ROCF VLM · 복합도형 그리기 과정으로 아밀로이드 베타 예측하기](challenges/rocf_vlm.md)(`CU + MCI`)로 통합되었습니다.** 인턴십 2단계 지원자는 L1 과제를 입문 연습으로 참고하되, 실제 제출은 ROCF VLM을 기준으로 [5절](#5-결과-정리와-공유)의 **비공개(Private) GitHub 저장소 및 메일 제출 절차**를 따라 주세요.

## 3. 준비물

- 노트북이나 데스크톱 컴퓨터. 모든 챌린지의 기본 실행과 검증은 GPU 없이 CPU만으로 돌아갑니다(ROCF VLM에서 딥러닝 실험을 할 때만 선택적으로 무료 Google Colab T4 16 GB 등을 쓰면 됩니다).
- **Python 3.11 또는 3.12.** 터미널에서 `python3 --version`으로 확인합니다. 없거나 버전이 다르면(예: macOS 기본 3.9, Ubuntu 22.04 기본 3.10) 3.12를 설치하고, 아래 ②에서 `python3` 대신 `python3.12`를 씁니다. macOS는 [python.org](https://www.python.org/downloads/)의 설치 파일을 씁니다. Linux는 배포판 패키지에 3.12가 있으면 그것을 쓰고, 없으면(Ubuntu 22.04 등) Ubuntu 24.04(기본 3.12)를 쓰는 것이 가장 간단합니다. Windows는 아래처럼 WSL의 Ubuntu 24.04를 씁니다.
- 터미널(macOS: 터미널 앱, Linux: 기본 터미널). 명령을 복사해 붙여 넣을 수 있으면 충분합니다.
- 인터넷 연결(패키지 설치와 실제 공개 데이터 다운로드용)과 디스크 여유 2 GB 정도. 실제 공개 데이터 단계는 받는 데 서버 상태에 따라 30초~6분이 걸리고 약 60 MB를 더 씁니다.

설치와 실행은 macOS(Apple 칩)와 Linux(ARM64)에서 확인했습니다. Windows에서는 확인하지 않았습니다. Windows라면 [WSL](https://learn.microsoft.com/ko-kr/windows/wsl/install)(Windows에서 Linux를 쓰는 기능)을 설치한 뒤 Linux 안내대로 하기를 권합니다. WSL에서는 Ubuntu 24.04(기본 Python 3.12)를 고르세요. Ubuntu 22.04의 기본 Python은 3.10이라 이 안내의 버전과 맞지 않습니다.

## 4. 설치와 첫 실행 (30분 안팎)

**① 저장소 내려받기.** GitHub의 이 저장소 첫 페이지에서 위쪽의 초록색 **Code** 버튼을 누르고 **Download ZIP**을 고른 뒤 압축을 풉니다. `git`을 쓸 줄 안다면 `git clone`으로 받아도 됩니다. 터미널에서 내려받은 폴더로 이동합니다.

```bash
cd 내려받은-폴더-이름     # ZIP으로 받았다면 보통 "저장소이름-main"
```

WSL에서는 ZIP 대신 WSL 터미널에서 바로 받는 편이 쉽습니다. 저장소 첫 페이지의 초록색 **Code** 버튼을 눌러 HTTPS 주소를 복사한 뒤 `git clone 복사한_저장소_URL`을 실행하면 WSL 홈 폴더에 저장소가 생기고, 설치와 실행도 빠릅니다(`git`이 없으면 `sudo apt install git`).

**② 가상환경 만들기.** 가상환경은 이 과제에 필요한 패키지를 컴퓨터의 다른 프로그램과 섞이지 않게 따로 담는 폴더입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

명령줄 앞에 `(.venv)`가 붙으면 성공입니다. 터미널을 새로 열 때마다 `source .venv/bin/activate`를 다시 실행합니다.

**③ 패키지 설치.** 먼저 EEG 챌린지의 패키지를 설치합니다. 몇 분 걸립니다.

```bash
pip install -r challenges/eeg/requirements.txt
```

fMRI 챌린지도 할 거라면 이어서 설치합니다. macOS, 일반 PC(x86_64)의 Linux·WSL에서는 미리 만들어진 설치 파일이 있어 바로 설치됩니다. Linux ARM64(예: 일부 서버, 라즈베리 파이)에서만 fMRI 패키지 brainiak을 직접 빌드하므로 C++ 컴파일러가 필요합니다(Ubuntu: `sudo apt install build-essential python3-dev`).

```bash
pip install -r challenges/fmri/requirements.txt
```

ROCF VLM 챌린지(L2)도 할 거라면 이어서 아래 패키지를 설치합니다.

```bash
pip install -r challenges/rocf_vlm/requirements.txt
```

**선택: `OPENROUTER_API_KEY` 환경변수 설정 (ROCF VLM 시각 탐색 시에만 필요).** OpenRouter API 키는 외부 시각-언어 모델(VLM)을 불러 그리기 과정 패널의 시각 패턴을 탐색해 보는 개발용 열쇠입니다. 기본 기준선(`starter_pipeline.py`)과 규격 점검(`smoke_test.py`)에는 **API 키가 전혀 필요 없으며**, 최종 제출용 `run_pipeline.py`도 반드시 외부 API 없이 오프라인(`OPENROUTER_API_KEY=""`)으로 끝까지 실행되어야 합니다. 개발 중에 `rocf_vlm_synthetic_example.ipynb`의 선택 셀로 VLM 탐색을 해 볼 때만 터미널에서 환경변수로 설정하세요(Windows PowerShell이면 `$env:OPENROUTER_API_KEY="발급받은_키"`).

```bash
export OPENROUTER_API_KEY="발급받은_키"
echo ${#OPENROUTER_API_KEY}   # 키 값 자체 대신 글자 수만 출력해 설정 여부 확인
```

> **API 키 보안 주의**: API 키 문자열을 `.py` 코드나 `.ipynb` 노트북 셀에 직접 적어 넣지 마세요. `.env` 파일을 만든 경우에도 `.gitignore`에 추가해 Git에 올라가지 않게 해야 합니다(비공개 저장소라도 키를 올리지 마세요).

**④ 데이터 만들고 노트북 열기.** EEG를 예로 듭니다.

```bash
cd challenges/eeg
python generate_synthetic_eeg.py
jupyter notebook eeg_synthetic_example.ipynb
```

`python generate_synthetic_eeg.py`의 마지막 줄에는 깜빡임 개수(정답의 일부)와 요약 통계 설정(기본값)이 함께 나옵니다. 미리 보고 싶지 않다면 "저장 위치" 줄까지만 확인하세요.

**실행 결과 확인.** 브라우저에 노트북이 열립니다. 열리지 않으면(WSL에서 흔함) 터미널에 나온 `http://localhost:8888/...token=...` 주소를 복사해 브라우저 주소창에 붙여 넣습니다. 위 메뉴의 **Run → Run All Cells**로 끝까지 실행해 봅니다. 마지막 부분에 "38회 중 36회"를 찾았다는 결과가 나오면 준비가 끝난 것입니다. 이제 [챌린지 문서](challenges/eeg.md)의 과제를 하나씩 해 봅니다.

**노트북이 켜져 있는 동안 다른 명령 치기.** `jupyter notebook`이 켜져 있는 동안에는 그 터미널에 다른 명령을 칠 수 없습니다. 다른 명령(다른 seed 데이터 만들기 등)은 새 터미널을 열고 저장소 맨 위 폴더로 `cd`해 `source .venv/bin/activate`를 실행한 뒤, 챌린지 폴더(`cd challenges/eeg`)로 이동해서 칩니다. 노트북 서버를 Ctrl+C로 끈 뒤 같은 터미널에서 해도 됩니다. 데이터를 만드는 방법은 챌린지 폴더의 `README.md`에 더 자세히 있습니다.

**막혔을 때**

| 증상 | 해결 |
| --- | --- |
| `python3: command not found` | Python을 설치하고 터미널을 다시 엽니다. |
| `python3 -m venv`가 `ensurepip` 오류로 실패 | Ubuntu·WSL에서는 `sudo apt install python3-venv`로 가상환경 기능을 설치한 뒤 다시 실행합니다. |
| `No module named 'encodings'` 또는 다른 프로젝트의 패키지가 섞여 보임 | 셸에 `PYTHONPATH`가 설정되어 있을 수 있습니다. `unset PYTHONPATH`를 실행한 뒤 `.venv` 폴더를 지우고 처음부터 다시 만듭니다. |
| `pip install`이 실패 | `python3 --version`이 3.11이나 3.12인지 확인합니다. 더 새 버전(3.13 이상)이나 옛 버전에서는 일부 패키지가 설치되지 않을 수 있습니다. Linux ARM64라면 C++ 컴파일러를 설치했는지(③) 확인합니다. |
| `No such file or directory: data/...` | L1(EEG·fMRI)은 노트북과 같은 폴더에서 `generate_synthetic_*.py`를 실행했는지, L2(ROCF VLM)는 `python -m zipfile -e data.zip .`로 압축을 풀었는지 확인합니다. |
| `python: command not found` | 가상환경이 꺼진 상태입니다. 저장소 맨 위 폴더에서 `source .venv/bin/activate`를 실행합니다. |
| `jupyter: command not found` | 가상환경이 켜져 있는지(`(.venv)` 표시) 확인합니다. |

## 5. 결과 정리와 공유

각 챌린지 문서의 **정리해 둘 것**을 만듭니다. 그림, 실행한 노트북·파이프라인, 짧은 설명입니다. 자율 학습과 2026 동계 인턴십 2단계 지원은 제출 방식이 완전히 다릅니다.

### (A) 오픈 챌린지 자율 학습자: 공개 GitHub + Issues 공유 (선택)

정답 파일로 스스로 확인할 수 있으므로 제출은 필수가 아닙니다. L1 연습 결과나 개인 학습 결과를 공유하고 의견을 받고 싶다면, 이 저장소의 **Issues** 탭에서 **New issue → 결과 공유하기**를 골라 내 결과가 있는 링크(본인 GitHub 저장소나 노트북 공유 링크)를 남겨 주세요. 링크가 없다면 GitHub에서 새 저장소를 만들고(**+ → New repository**), **Add file → Upload files**로 실행한 노트북(`.ipynb`)과 그림만 올린 뒤 그 주소를 적으면 됩니다. GitHub는 올린 노트북을 화면에 그대로 보여 줍니다.

이슈는 누구나 볼 수 있으므로 개인정보(전화번호, 학번, 메일)는 적지 마세요. 이름도 적지 않아도 됩니다. 실행한 노트북을 공유하기 전에 출력에 내 사용자 이름이 든 컴퓨터 경로가 남아 있지 않은지 확인하세요. EEG 실제 데이터 노트북은 처음 받을 때 1단계 출력에 저장 경로를 보여 줍니다. 노트북에서 내 사용자 이름을 검색해 보고, 있으면 노트북을 한 번 더 실행하면 사라집니다. 그림과 숫자는 공유해도 되지만, 받은 데이터 파일은 올리지 마세요. `git`으로 올릴 때는 `.gitignore`가 `data`, `data_*` 폴더를 막아 주지만, 웹에서 **Upload files**로 올릴 때는 막아 주지 않으니 이 폴더들을 고르지 마세요.

### (B) 2026 동계 인턴십 2단계 지원자: Git 기본 사용법 + 비공개(Private) GitHub 저장소 제출 (필수)

> **공개 업로드 절대 금지**: 2026 동계 인턴십 2단계(`ROCF VLM`) 지원자는 **공개(Public) GitHub 저장소나 이 저장소의 공개 Issues에 지원 과제 코드·풀이·결과·비공개 저장소 주소를 절대 올리지 마세요.**

1. **비공개(Private) GitHub 저장소 만들기**: GitHub 로그인 후 오른쪽 위 **`+` → New repository**를 누릅니다. 저장소 이름을 적고, 공개 설정에서 반드시 **Public**이 아닌 **Private**을 고른 뒤 **Create repository**를 누릅니다.
2. **`.gitignore` 점검 후 내 비공개(Private) 저장소로 올리기**: 큰 합성 데이터나 API 키 파일이 저장소에 올라가지 않도록 저장소 루트의 `.gitignore`를 확인합니다(기본 `.gitignore`의 `data/`, `data_*/`, `.venv/`에 더해 `.env`, `outputs*/`, `submissions/`도 함께 적어 두는 것을 권장합니다). 그다음 저장소 루트 폴더에서 4절 ①의 내려받기 방식에 맞는 순서를 실행합니다.

   - **(경우 1) `4절 ①`에서 `git clone`으로 받은 경우**: 이미 `origin`이 공개 저장소를 가리키고 있으므로 `git remote add origin`을 쓰면 오류가 납니다. 대신 `git remote set-url origin`으로 내 비공개 저장소 주소로 바꾼 뒤 `git remote -v`로 확인하고 올립니다.

```bash
git add .gitignore challenges/rocf_vlm/
git status                                # 스테이징된 파일 확인 (data/, .env, 가중치 파일이 없는지 점검)
git commit -m "Add ROCF VLM pipeline"
git branch -M main
git remote set-url origin 복사한_내_Private_저장소_HTTPS_URL
git remote -v                             # origin이 반드시 내 Private 저장소 주소를 가리키는지 확인
git push -u origin main
```

   > **주의**: `git remote -v` 결과에 여전히 공개 저장소 주소가 보인다면 **절대 `git push`를 실행하지 말고**, 위 `git remote set-url origin ...` 명령을 먼저 실행해 내 Private 저장소 주소로 바뀐 것을 확인한 뒤 `git push` 하세요.

   - **(경우 2) `Download ZIP`으로 받았거나 새 폴더에서 시작한 경우**: `.git` 폴더가 없으므로 `git init`부터 실행합니다(빈 새 폴더에서 시작한다면 먼저 압축을 푼 폴더의 `.gitignore`와 `challenges/rocf_vlm/` 폴더를 복사해 둔 뒤 실행하세요).

```bash
git init
git add .gitignore challenges/rocf_vlm/
git status                                # 스테이징된 파일 확인 (data/, .env, 가중치 파일이 없는지 점검)
git commit -m "Add ROCF VLM pipeline"
git branch -M main
git remote add origin 복사한_내_Private_저장소_HTTPS_URL
git remote -v                             # origin이 내 Private 저장소 주소를 가리키는지 확인
git push -u origin main
```

   GitHub에서 `git push` 비밀번호를 물으면 계정 로그인 비밀번호 대신 GitHub **Personal Access Token(PAT)** 또는 `gh auth login` 인증을 사용합니다.
3. **채점 계정 초대하기**: 만든 비공개 저장소 페이지에서 **Settings → Collaborators → Add people**을 눌러 `11/25(수)` 2단계 안내 메일에 적힌 채점용 GitHub 계정을 초대해 주세요(메모: 초대를 수락해야 채점자가 코드를 읽을 수 있으므로 마감 전에 초대가 완료되었는지 확인합니다).
4. **2단계 체크포인트와 메일 제출**:
   - **1차 중간 제출 (`12/1(화) 23:59`)**: 사전 확정 확장 질문·반증 기준과 기준선 커밋 해시를 메일 제목 `[2026동계인턴챌린지-기준] 지원번호`로 제출합니다. 기준선 커밋 해시는 저장소 루트에서 `git rev-parse --short HEAD` 명령으로 출력되는 7자리 커밋 ID를 복사하거나, GitHub 내 Private 저장소의 커밋 이력(`Commits`)에서 해당 커밋 오른쪽에 표시된 7자리 ID를 복사하면 됩니다.
   - **최종 제출 (`12/7(월) 23:59`)**: 메일 제목 `[2026동계인턴챌린지] 지원번호_ROCF`로 (1) 채점 계정이 초대된 **Private GitHub 저장소 링크**, (2) **A4 2쪽 이내 보고서 PDF**, (3) **A4 1쪽 이내 AI 활용 기록** 3종을 제출합니다. 자세한 규격·일정은 [ROCF VLM 챌린지 안내](challenges/rocf_vlm.md)를 확인해 주세요.

## 6. 질문하기

막힌 곳이나 궁금한 점은 **Issues** 탭에서 **New issue → 질문하기**를 골라 남겨 주세요. 문서가 실제 결과와 다르면 **문서 오류 제보**를 골라 주세요. 어떤 명령을 실행했고 어떤 메시지가 나왔는지 그대로 붙여 넣으면 답하기 쉽습니다. 다만 출력에 들어 있는 내 컴퓨터 경로의 사용자 이름과 API 키(`OPENROUTER_API_KEY`)는 지우고 붙여 넣어 주세요. 이슈는 누구나 볼 수 있는 **공개 공간**이므로 **인턴십 2단계 지원자는 과제 코드, 풀이 전략, 비공개 저장소 주소, 지원번호를 절대 적지 마세요.** 답변은 연구실 구성원이 가능한 범위에서 드리며, 응답 시간은 보장하지 않습니다. 이슈와 댓글에서는 [행동 규칙](CODE_OF_CONDUCT.md)을 지켜 주세요.

## 7. 자주 묻는 질문

**배경지식이 없어도 되나요?** L1 챌린지는 필요한 용어를 문서와 노트북에서 설명하므로 Python 문법을 조금 알면 충분합니다. L2 챌린지(ROCF VLM)는 교차검증이나 AUROC 같은 기초 머신러닝 개념을 알면 수월하며, 기본 파이프라인과 아래 8절 용어 설명이 함께 제공되어 구조를 보며 시작할 수 있습니다.

**AI 도구(ChatGPT, Claude 등)를 써도 되나요?** 써도 됩니다. 오픈 챌린지 자율 학습 결과를 Issues에 공유할 때는 어떤 부분을 AI에게 맡겼는지 양식에 적어 주세요. 인턴십 2단계 지원자는 별도 안내된 **A4 1쪽 이내 AI 활용 기록**으로 제출하며, 이 챌린지의 핵심은 내가 직접 코드를 이해하고 설명할 수 있는 것입니다.

**비용이 드나요?** 없습니다. 모든 코드와 데이터는 무료이고, 노트북 컴퓨터에서 돌아갑니다(인턴십 2단계 지원자는 선택적 VLM 탐색용 OpenRouter 키도 별도 안내 범위에서 지원됩니다).

**채용이나 인턴 선발과 연결되나요?** 오픈 챌린지 자체는 스스로 연구 적합성을 가늠해 보는 학습 기회입니다. 2026 동계 인턴십 선발 2단계 연구 챌린지는 **ROCF VLM**(`CU + MCI`)으로 통합 운영되며, 지원자는 공개 Issues가 아니라 [5절 (B)](#5-결과-정리와-공유)의 비공개 저장소 및 메일 절차로만 제출합니다.

## 8. 용어

- **합성 데이터**: 실제 사람에게서 잰 것이 아니라, 정해진 특성을 갖도록 컴퓨터로 만든 데이터.
- **아티팩트**: 측정하려는 대상(뇌) 외의 원인(눈, 근육, 머리 움직임)으로 기록에 섞여 든 신호.
- **seed**: 무작위 생성의 시작값. 같은 seed면 항상 같은 데이터가 나옵니다.
- **가상환경(venv)**: 한 과제에 필요한 Python 패키지를 따로 담아 두는 폴더.
- **노트북(Jupyter)**: 코드, 그림, 설명을 한 문서에서 차례로 실행하는 파일(`.ipynb`).
- **상관계수**: 두 신호가 함께 오르내리는 정도. −1에서 1 사이 값.
- **교란 변수(Confounding)**: 예측하려는 대상(예: 아밀로이드 베타 양성 여부)과 입력 특징 양쪽에 모두 연결되어 있어, 실제 관계가 없는데도 모델이 잘 맞히는 것처럼 착시를 일으키는 제3의 변수(예: `MCI` 진단 여부, fMRI의 머리 움직임).
- **천장효과(Ceiling effect)**: 과제가 특정 집단에게 쉬워 점수가 만점 근처에 몰리면서, 점수나 정적 그림만으로는 그 집단 내부의 차이를 구분하기 어려워지는 현상.
- **ROCF(복합도형 검사)**: 복잡한 기하 도형(18개 요소, 만점 36점)을 보고 따라 그린 뒤(`Copy`) 기억에 의존해 다시 그리는(`Delayed Recall`) 신경심리검사.
- **CU(인지정상) / MCI(경도인지장애)**: `CU`는 인지 기능이 정상 범위인 집단, `MCI`는 객관적 인지 저하가 시작된 집단입니다.
- **아밀로이드 베타 / Centiloid**: 알츠하이머병 병리와 관련된 뇌 단백질(아밀로이드 베타)과 그 뇌 축적량을 여러 영상 기준에 걸쳐 0–100 안팎의 표준 척도로 맞춘 연속 지표(Centiloid, 약 25 CL 안팎을 양성 기준으로 둠).
- **AUROC**: 모델이 양성(예: `Aβ+`)과 음성(`Aβ-`)을 얼마나 잘 구분하는지 나타내는 지표. `0.50`은 동전 던지기(무작위), `1.00`은 완벽한 분류입니다.
- **교차검증(Cross-Validation) / 홀드아웃(Holdout)**: 학습 데이터를 여러 조각(폴드)으로 나누어 번갈아 검증해 일반화 성능을 재는 방법이 교차검증이고, 모델 선택 과정에서 끝까지 보지 않고 따로 떼어 둔 시험 표본이 홀드아웃(`test`)입니다.
- **특권 정보 활용 학습(Learning Using Privileged Information)**: 학습(`train`) 단계에서는 뇌영상·임상 지표 등 풍부한 특권 정보를 함께 써서 모델을 훈련하되, 실제 평가(`test`) 단계에서는 저비용 입력(ROCF 그림·펜 궤적)만으로 예측하게 하는 학습 방식입니다.
- **반증 대조군(Falsification control)**: 모델이 진짜 신호가 아니라 가짜 단서(교란·우연)에 기댔는지 걸러 내기 위해 일부러 핵심 과정 단서를 제거하거나 순서를 교란해 보는 대조 실험.
- **VLM(시각-언어 모델) / API 키(`OPENROUTER_API_KEY`)**: 그림과 텍스트 질문을 함께 읽고 답하는 AI 모델이 VLM이며, 코드에서 외부 VLM 서버를 호출할 때 쓰는 개인 인증 문자열이 API 키입니다. 코드나 Git에 절대 올리면 안 됩니다.
- **Git / 비공개(Private) 저장소**: 코드 변경 이력을 기록·공유하는 도구가 Git이고, GitHub에서 나(와 초대한 채점자)만 볼 수 있게 잠근 코드 공간이 **Private** 저장소입니다(`Public`은 전 세계에 공개됩니다).

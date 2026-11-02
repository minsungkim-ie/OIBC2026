# OIBC2026
2026 OIBC 경진대회 전국의 발전소를 묶어 내일의 발전량과 수익을 예측할 참가자를 모집합니다.

태양광 발전소 253개소를 두 집합자원(G1, G2)으로 나누고, 각 집합의 시간별 발전량을 하루 전에 예측한다. 마감 2026-11-13 23:59:59 KST, 제출 총 20회.

## 시작하기
```bash
git clone https://github.com/minsungkim-ie/OIBC2026.git
cd OIBC2026
uv sync
```
대회 원본 데이터는 git에 없다. 대회 사이트에서 받아 `handover 2/01_participant_dataset/` 구조로 저장소 폴더에 넣는다.

## 실행
```bash
uv run python scripts/run_all.py                              # 01 검증 → 02 구성 최적화 → 03 최종 학습·제출 파일 → 04 형식 검증
uv run python scripts/04_check_submission.py submissions/v1   # 제출 전 형식 검증
```

## 구조
| 경로 | 내용 |
|---|---|
| `docs/대회_규정.md` | 대회 개요·채점·데이터 규정·제출 규격·마감 (일반 사항은 여기 한 곳) |
| `docs/v1_방법론.md` | v1 방법 설명(전처리·피처·모델·구성 최적화)과 검증 결과 |
| `EXPERIMENTS.md` | **제출 이력·점수·다음 후보** (작업 전 먼저 읽기) |
| `CLAUDE.md` | AI 어시스턴트·팀 작업 규칙 (절대 규칙, 버전·협업 규칙) |
| `src/oibc/`, `scripts/` | 코드 (설정은 `src/oibc/config.py`) |
| `submissions/vN/` | 제출한 파일 (수정하지 않음) |
| `outputs/` | 검증·구성 리포트(md) |

## 버전·협업 규칙
- **버전 번호 = 대회 사이트에 낸 제출 순서** (v1, v2, …). 제출 파일은 `submissions/vN/`, 코드는 `git tag vN`으로 고정.
- `main`에는 제출했거나 제출 후보인 코드만. 실험은 `exp/<이름>` 브랜치에서 하고 점수가 오른 것만 합친다.
- 작업 전 `git pull`, 후 `git push`. 같은 파일을 동시에 고치지 않도록 역할을 나눈다.

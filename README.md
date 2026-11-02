# OIBC2026
2026 OIBC 경진대회 전국의 발전소를 묶어 내일의 발전량과 수익을 예측할 참가자를 모집합니다.

## 시작하기
```bash
git clone https://github.com/minsungkim-ie/OIBC2026.git
cd OIBC2026
uv sync
```
대회 원본 데이터(`handover 2/01_participant_dataset/`)는 git에 없다. 대회 사이트에서 받아 저장소 폴더 안에 `handover 2/` 그대로 넣는다.

## 실행
```bash
uv run python scripts/run_all.py          # 01 검증 → 02 구성 최적화 → 03 최종 학습·제출 파일 → 04 형식 검증
uv run python scripts/04_check_submission.py submissions/v1   # 제출 전 형식 검증
```

## 버전 규칙 (제출 총 20회)
- **버전 번호 = 대회 사이트에 낸 제출 순서.** 낼 때마다 v1 → v2 → … 로 하나씩 올린다.
- 제출 파일: `submissions/vN/` (한 번 낸 폴더는 수정하지 않음) / 코드 고정: `git tag vN`
- 점수·변경점 기록: [`EXPERIMENTS.md`](EXPERIMENTS.md) — 작업 전에 먼저 읽는다.
- 대회 전체 규정(마감 11-13, 제출 규격, 허용 데이터): [`docs/대회_컨텍스트.md`](docs/대회_컨텍스트.md)
- 방법 설명: `설명서_v1_방법론.md`, 규칙: [`CLAUDE.md`](CLAUDE.md)

## 협업
- `main`에는 제출했거나 제출 후보인 코드만. 실험은 `exp/<이름>` 브랜치에서 하고 점수가 오른 것만 합친다.
- 작업 전 `git pull`, 후 `git push`. 같은 파일을 동시에 고치지 않도록 역할을 나눈다.

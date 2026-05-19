# HumTune - Design Overview

## 0. 사용 지침

이 설계는 아래 원칙을 반드시 유지해야 한다.

- 설계를 변경하지 말 것
- AI 역할을 확대하지 말 것
- 규칙 기반 파이프라인을 제거하지 말 것
- MVP 범위를 확장하지 말 것

이 문서를 기반으로 작업할 때는 기존 Spring orchestration 구조와 deterministic pipeline 철학을 유지한다.

---

## 1. 프로젝트 개요

### 한 줄 정의

사용자의 허밍을 ML 기반으로 note event로 전사한 뒤,  
결정적 후처리로 멜로디를 정리하고 간단한 코드와 피아노 반주를 생성하는 음악 보조 시스템

---

## 2. 핵심 설계 철학

### 2.1 Deterministic Pipeline

동일 입력 → 동일 결과를 보장해야 한다.

오디오 입력 → ML transcription → deterministic cleanup → 규칙 기반 처리 → MIDI 생성

ML은 원시 오디오에서 note event 후보를 얻기 위한 전사 도구로 사용한다.  
scale fitting, quantization, chord inference, MIDI 생성은 시스템 코드의 결정적 규칙으로 처리한다.

---

### 2.2 AI 역할 (최종 정의)

AI는 음악을 생성하지 않는다.  
AI는 rule 기반 결과를 **설명, 평가, 보조 판단**하는 역할만 수행한다.

AI의 역할:

- 전사 및 후처리 결과에 대한 사용자 피드백 생성
- 코드 및 결과 설명 생성
- 사용자 허밍 피드백 생성
- 생성된 결과의 자연스러움 평가
- chord/scale 후보가 애매한 경우 보조 평가
- 반복 실패 패턴 또는 품질 문제를 설명하는 평가 문구 생성

---

### 2.3 AI 금지 영역

AI는 절대 수행하지 않는다:

- note transcription 직접 결정
- 음정 보정
- 박자 보정
- scale/chord 단독 결정
- MIDI 생성

→ 모든 생성 로직은 시스템 코드가 담당한다

AI는 Basic Pitch 결과나 deterministic cleanup 결과를 바꾸지 않는다.

---

## 3. MVP 범위

### 포함

- 허밍 업로드 (5~10초)
- Basic Pitch 기반 note transcription 평가 및 적용
- note event 정리
- scale fitting
- quantization
- chord 생성
- 단순 피아노 반주 MIDI 생성
- 결과 재생

### 제외

- 완성곡 생성
- 보컬 합성
- 다중 악기 편곡
- 실시간 처리
- AI 기반 멜로디/코드 생성
- 복잡한 반주 편곡

---

## 3.1 ML Transcription 전략

현재 병목은 raw pitch extraction 자체보다, 허밍을 음악적으로 유효한 note event로 전사하는 품질이다.

기존 librosa.pyin + custom segmentation 중심 접근은 MVP에서 튜닝 비용이 크다.  
HumTune은 Basic Pitch를 note transcription 단계의 기본 엔진으로 사용한다.

HumTune의 경쟁 가치는 직접 pitch/segmentation 알고리즘을 만드는 데 있지 않다.  
핵심 가치는 다음에 있다:

- ML transcription 결과를 서비스 흐름에 안정적으로 orchestration
- note event normalization
- deterministic cleanup
- scale fitting 및 quantization
- chord inference
- 사용자 피드백 및 실패 처리
- 반복 가능한 결과와 신뢰성

Basic Pitch 출력은 최종 결과가 아니라 후속 규칙 기반 파이프라인의 입력이다.

deterministic cleanup은 Basic Pitch raw note event에 대해 다음을 수행한다:

- 너무 짧거나 약한 transient note 제거
- 짧은 chromatic pitch transient / neighbor note 제거
- 같은 pitch의 과도한 fragment 병합
- 충분한 onset 간격과 note 길이를 가진 same-pitch 반복은 articulation으로 보존
- 동시에 겹치는 후보 중 안정적인 단일 melody 후보 선택
- 의미 있는 repeated note onset 보존
- overlap은 뒤 note를 밀지 않고 앞 note를 줄여 onset을 보존

---

## 4. 실행 모델 (확정)

### 4.1 비동기 처리

POST /api/audio

1. 파일 저장
2. AnalysisRequest 생성 (PENDING)
3. 즉시 응답 반환

이후:

- async worker 실행
- PROCESSING 전환
- Python Audio Service 호출

---

### 4.2 상태 전이

- PENDING → PROCESSING → COMPLETED
- PENDING → PROCESSING → FAILED

---

### 4.3 요청-응답 규칙

- API는 Python 분석을 기다리지 않는다
- 결과는 polling으로 조회한다

---

## 5. 상태 모델

- PENDING
- PROCESSING
- COMPLETED
- FAILED

---

## 6. 핵심 처리 흐름

Upload  
→ AnalysisRequest 생성 (PENDING)  
→ Async worker 실행  
→ PROCESSING  
→ Python 분석  
→ Basic Pitch  
→ note events  
→ deterministic cleanup  
→ scale fitting  
→ quantization  
→ chord inference  
→ MIDI  
→ Spring orchestration  
→ AI 평가 및 피드백 생성  
→ 결과 저장  
→ COMPLETED

실패 시:

→ FAILED  
→ errorMessage 저장

---

## 7. Deterministic 규칙

### 7.1 Scale 선택

- major/minor 후보 생성
- note distance 합 최소 선택

tie-break:

1. scale tone 포함 비율
2. phrase anchor 점수 (첫음/끝음의 tonic, dominant, mediant 여부 및 최저음 tonic 여부)
3. tonic 포함 여부
4. C Major

---

### 7.2 Tempo

- pitch 간 시간 간격 기반 추정

fallback:

- 100 BPM

---

### 7.3 Quantization

- 1/8 note grid
- cleaned melody phrasing을 baseline으로 사용
- onset은 grid에 충분히 가까운 경우에만 보수적으로 snap
- duration은 다음 onset까지 강제로 늘리지 않고 note 자체 길이를 우선 quantize
- repeated note articulation은 note count와 onset 순서를 보존

---

### 7.4 Chord 선택

- diatonic chord 후보
- window 단위 melody 포함 길이 scoring
- chord window는 0초 고정이 아니라 melody 시작 시점부터 phrase-relative하게 배치
- 짧은 허밍 melody는 3~4개 chord section을 우선 사용
- window 시작 melody note가 chord root와 일치하면 작은 anchor bonus 적용
- 첫 chord는 안정적인 진입을 위해 tonic을 우선
- 단순 동요형 멜로디에서는 과도한 전환보다 안정적인 진행을 우선하되, 비화성음을 강제로 생성하지 않음

tie-break:

1. tonic
2. dominant
3. subdominant

---

### 7.5 Progression

- 3~4 chord
- 시작: tonic
- 종료: tonic 또는 dominant

---

## 8. 실패 처리

### 8.1 Transcription 실패

→ Basic Pitch 실패 또는 유효 note event 부족 시 재시도  
→ 실패 시 단일 note melody

---

### 8.2 Note 실패

→ default melody 생성  
→ COMPLETED 유지

---

### 8.3 Chord 실패

→ 기본 progression

C - F - G - C

---

### 8.4 MIDI 실패

→ FAILED

---

### 8.5 AI 실패

→ COMPLETED 유지  
→ feedbackText = null

---

## 9. Python Audio Service 계약

### Request

POST /internal/audio/analyze

- audioId
- rawAudioPath

---

### Response

성공:

- status: COMPLETED
- detectedScale
- keyConfidence
- originalNotes
- adjustedNotes
- chords
- midiPath
- previewAudioPath
- processingTimeMs

실패:

- status: FAILED
- errorMessage

성공 응답 필드의 의미:

- detectedScale: 최종 scale fitting 결과. 이후 scale adjustment, quantization, chord inference에 사용된 scale 이름
- keyConfidence: 선택된 detectedScale에 대한 deterministic confidence score
- originalNotes: Basic Pitch raw note event에서 추출한 원본 note name sequence. 사용자 표시용 최종 melody가 아니라 진단 및 호환용 필드
- adjustedNotes: legacy-compatible field name. API가 노출하는 최종 melody note name sequence이며 cleanup, scale adjustment, quantization이 모두 적용된 melody이다. midiPath의 melody track과 의미적으로 일치해야 한다
- chords: 최종 chord inference 결과의 chord label sequence만 노출한다. chord startTime/duration은 API 응답에 포함하지 않는다
- midiPath: 서비스의 main product MIDI. 최종 melody와 timing-normalized inferred chord/accompaniment를 포함한다
- previewAudioPath: midiPath에서 렌더링한 선택적 WAV preview 경로. preview 생성이 실패하면 생략될 수 있다
- processingTimeMs: Python 분석 처리 시간

API 응답에는 raw/cleaned/scale-adjusted-before-quantization/final melody 전체를 모두 노출하지 않는다.  
raw, cleaned, adjusted, final melody, chord-only, combined MIDI 비교 산출물은 manual/debug artifact로만 유지한다.

---

### 입력 조건

- Spring이 파일 존재 보장
- Python이 읽기 검증
- 내부에서 오디오 표준화 수행

---

### 재실행 규칙

- 동일 audioId → overwrite 허용
- 결과는 deterministic

---

## 10. AI Assistant 계약

### 입력

- detectedScale
- adjustedNotes
- chords

---

### 출력

- feedbackText
- chordExplanation
- naturalnessScore

---

### 원칙

- AI는 결과를 생성하지 않는다
- AI는 설명, 피드백, 평가만 수행한다
- AI는 note, scale, chord, MIDI를 직접 수정하지 않는다
- AI 실패는 전체 실패가 아니다

---

## 11. Timeout / 장애 처리

- Python 호출 timeout: 15초
- timeout 시 FAILED

- PROCESSING 최대: 60초

### 판정 주체

- Spring async worker

### 처리

- 60초 초과 시 FAILED

---

## 12. API 구조

POST /api/audio  
GET /api/audio/{audioId}  
GET /api/audio/{audioId}/result
GET /api/audio/{audioId}/files/preview
GET /api/audio/{audioId}/files/midi

---

## 13. 저장 구조

### DB

- audio_meta
- analysis_request
- analysis_result

---

### 파일

- raw audio
- MIDI
- preview audio

---

## 14. 아키텍처

Frontend  
→ Spring Boot  
→ Python Audio Service  
→ Local Storage  
→ PostgreSQL

---

## 15. 성능 기준

- 10초 이내 결과

---

## 16. 테스트 기준

- 정상 입력
- 잡음 입력
- pitch 실패
- chord 실패
- AI 실패
- timeout

---

## 17. 절대 변경 금지

1. AI가 음악 생성 금지
2. rule 기반 제거 금지
3. pipeline 단순화 금지
4. MVP 확장 금지

---

## 최종 요약

- deterministic pipeline
- rule-based generation
- async processing
- controlled failure
- AI = 설명 + 평가

# HumTune - Design Overview

## 0. 사용 지침

이 설계는 아래 원칙을 반드시 유지해야 한다.

- 설계를 변경하지 말 것
- AI 역할을 확대하지 말 것
- 규칙 기반 파이프라인을 제거하지 말 것
- MVP 범위를 확장하지 말 것

이 문서를 기반으로 작업할 때는 “수정”이 아니라 “보완”만 허용된다.

---

## 1. 프로젝트 개요

### 한 줄 정의

사용자의 허밍을 분석하여 멜로디를 보정하고,  
어울리는 코드와 피아노 반주를 생성하는 AI-assisted 음악 보조 시스템

---

## 2. 핵심 설계 철학

### 2.1 Deterministic Pipeline

동일 입력 → 동일 결과를 보장해야 한다.

오디오 입력 → 분석 → 규칙 기반 처리 → MIDI 생성

---

### 2.2 AI 역할 (최종 정의)

AI는 음악을 생성하지 않는다.  
AI는 rule 기반 결과를 **설명, 평가, 보조 판단**하는 역할만 수행한다.

AI의 역할:

- 코드 및 결과 설명 생성
- 사용자 허밍 피드백 생성
- 생성된 결과의 자연스러움 평가
- chord/scale 후보가 애매한 경우 보조 평가

---

### 2.3 AI 금지 영역

AI는 절대 수행하지 않는다:

- pitch 추출
- 음정 보정
- 박자 보정
- scale/chord 단독 결정
- MIDI 생성

→ 모든 생성 로직은 시스템 코드가 담당한다

---

## 3. MVP 범위

### 포함

- 허밍 업로드 (5~10초)
- pitch detection
- note sequence 변환
- scale fitting
- quantization
- chord 생성
- MIDI 생성
- 결과 재생

### 제외

- 완성곡 생성
- 보컬 합성
- 다중 악기 편곡
- 실시간 처리

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
→ Rule 기반 결과 생성  
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
2. tonic 포함 여부
3. C Major

---

### 7.2 Tempo

- pitch 간 시간 간격 기반 추정

fallback:

- 100 BPM

---

### 7.3 Quantization

- 1/8 note grid
- nearest grid

---

### 7.4 Chord 선택

- diatonic chord 후보
- melody 포함 비율 scoring

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

### 8.1 Pitch 실패

→ 재시도 → 실패 시 단일 note melody

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
- adjustedNotes
- chords
- midiPath
- previewAudioPath

실패:

- status: FAILED
- errorMessage

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
- AI는 설명과 평가만 수행한다
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

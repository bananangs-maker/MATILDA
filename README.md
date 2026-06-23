# 내 전용 차트 — 레버리지 ETF (Render 배포판)

TQQQ·SOXL 시세를 받아 캔들 차트 + 지표 + 티어 기반 매매 의사결정 패널을 보여주는 웹앱.

- 데이터: **Twelve Data**(무료, 클라우드에서 안정적) → 실패 시 Stooq → 합성(MOCK)
- 서버가 HTML·데이터를 같은 도메인에서 서빙 → CORS 없음
- 지연/일봉 데이터로 충분 (볼린저·이평 기반 판단)

## 데이터 키 발급 (필수 · 무료)

Render 같은 클라우드는 Yahoo/Stooq 직접 스크래핑이 차단되므로 정식 무료 API를 씁니다.

1. https://twelvedata.com 가입 (무료)
2. 대시보드에서 **API Key** 복사 (무료 800회/일)
3. 이 키를 Render 환경변수 `TWELVEDATA_API_KEY` 에 넣습니다 (아래 배포 단계 참고)

## GitHub → Render 배포

1. `my_chart` 파일들을 GitHub repo에 올림
2. Render → **New → Blueprint** → repo 선택 → `render.yaml` 자동 인식 → **Deploy**
3. 배포 후 **Render 서비스 → Environment** 에서:
   - `TWELVEDATA_API_KEY` = (발급받은 키)
   - `USE_MOCK` = `false`  (기본값)
   - 저장하면 자동 재배포
4. `https://<이름>.onrender.com/health` 가 `{"ok": true}` 면 정상

> 코드를 고쳐서 다시 올리면(같은 파일명으로 업로드/푸시) Render가 자동 재배포합니다.

## 알아둘 점

- **콜드 스타트**: 무료 플랜은 15분 무접속 후 슬립 → 다음 접속 시 30~50초 기동(정상).
- **호출 절약**: 서버가 종목별 10분 캐시를 둬서 무료 한도(800/일)를 거의 안 씁니다.
- **키 없이 화면만 확인**: Render 환경변수에서 `USE_MOCK=true` 로 두면 합성 데이터로 동작.

## 로컬 실행 (선택)
```bash
pip install -r requirements.txt
echo "TWELVEDATA_API_KEY=발급받은키" > .env
echo "USE_MOCK=false" >> .env
python app.py        # http://localhost:8000
```

## 자동 vs 수동 입력
| 자동 (차트·서버 계산) | 수동 (직접 입력) |
|---|---|
| 볼린저·EMA·SMA·SAR·거래량 | VIX |
| RSI·MFI·CCI·MACD | 공포·탐욕 지수 |
| 멀티이평·180선·장기크로스·다이버전스 | 피보나치 / 추세선 위치 |

## 파일
| 파일 | 역할 |
|---|---|
| `app.py` | Flask 앱 (대시보드 + /chart, 캐시, gunicorn 진입점 `app:app`) |
| `public_data.py` | 시세 수집 (Twelve Data → Stooq) |
| `indicators.py` | OHLCV → 차트 시리즈 + 의사결정 신호 |
| `dashboard.html` | 차트 UI + 의사결정 패널 |
| `render.yaml` | Render Blueprint 설정 |

> 면허 있는 투자 자문이 아니며, 본 도구는 사용자의 매매 규칙을 정리·자동화하는 보조 장치입니다.
> 투자 판단과 결과의 책임은 전적으로 사용자에게 있습니다.


## v2 추가 기능

- **백테스트**: 차트 하단. MACD 골든/데드크로스 + 추세선(SMA) 규칙을 과거 데이터로 검증. 전략 수익률 vs 매수후보유, 승률, 매매 횟수, 최대 낙폭, 자산곡선 표시. 추세선 SMA(50/120/200) 선택 가능.
- **매매 신호 마커**: 차트에 매수(▲)·매도(▼) 화살표. "매매신호" 토글로 on/off.
- **CNN 시장 심리**: 우측 "시장 심리" 카드 ↻ 갱신 → 공포·탐욕(자동 입력), 풋콜 옵션, VIX 변동성(CNN 정규화 점수). 원시 VIX는 Twelve Data 키 있을 때만 자동, 없으면 수동.
- **매수/매도 압력(OBV)**: 오실레이터 탭에 OBV 추가 + 심리 카드에 최근 20일 매수압력% (거래량 기반 프록시).

> 주의: 풋콜·VIX 컴포넌트는 CNN의 0~100 정규화 점수이며, 압력은 호가 데이터가 아닌 거래량 프록시입니다.
> 차트 패턴 자동인식·엘리엇 파동은 신뢰도 문제로 미포함입니다.


## v3 — 그리기 도구 + 저장

차트 위 툴바: **커서 / 수평선(지지·저항) / 추세선 / 마지막 삭제 / 전체 삭제**.
- 수평선: 도구 선택 후 차트를 한 번 클릭 → 그 가격에 지지/저항선.
- 추세선: 도구 선택 후 시작점·끝점 두 번 클릭 → 2점 추세선.
- 우측 "그리기 목록"에서 개별 삭제 가능. 종목(TQQQ/SOXL)별로 따로 저장됩니다.

### 저장 방식 (2단계)
1. **기본(설정 불필요)**: 그린 선은 이 **브라우저(localStorage)**에 저장 → 한 기기에선 트레이딩뷰처럼 유지.
2. **Supabase 연동(선택, 기기 간 동기화)**: 아래만 하면 자동으로 클라우드 저장으로 승격됩니다.

### Supabase 켜는 법
1. Supabase 프로젝트 → SQL Editor에서 테이블 생성:
   ```sql
   create table if not exists chart_drawings (
     ticker text primary key,
     drawings jsonb default '[]'::jsonb,
     updated_at timestamptz default now()
   );
   ```
2. Supabase → Project Settings → API 에서 **Project URL**과 **service_role 키** 복사.
3. Render → 서비스 → Environment 에 추가:
   - `SUPABASE_URL` = 프로젝트 URL (예: https://xxxx.supabase.co)
   - `SUPABASE_SERVICE_KEY` = service_role 키
4. 저장 → 자동 재배포. 이후 그린 선이 Supabase에 저장되어 다른 기기에서도 보입니다.

> service_role 키는 **서버(Render)에만** 두고 브라우저엔 노출하지 않습니다(이 앱이 그렇게 설계됨). 키는 외부 공개 금지.
> Supabase 미설정이거나 오류 시 자동으로 localStorage로 폴백하므로 그리기는 항상 작동합니다.

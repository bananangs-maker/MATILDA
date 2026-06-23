# 내 전용 차트 — 레버리지 ETF (Render 배포판)

TQQQ·SOXL 공개 시세를 받아 캔들 차트 + 지표(볼린저·EMA·SAR·거래량 + RSI/MACD/MFI/CCI) +
티어 기반 매매 의사결정 패널을 보여주는 웹앱. Render에 올리면 URL만으로 접속됩니다.

- **API 키·증권사 계좌 불필요** (공개 시세만 사용)
- 서버가 HTML·데이터를 같은 도메인에서 서빙 → **CORS 없음**
- 데이터: Yahoo Finance → 실패 시 Stooq 자동 폴백 → 둘 다 막히면 합성(MOCK)

---

## A. GitHub에 올리기

```bash
cd my_chart
git init
git add .
git commit -m "my chart"
# GitHub에서 빈 repo 생성 후:
git remote add origin https://github.com/<본인>/my-chart.git
git branch -M main
git push -u origin main
```

`.gitignore`가 `.env`를 제외하므로 비밀값이 올라갈 일은 없습니다(이 앱은 키 자체가 없음).

## B. Render에서 배포

### 방법 1 — Blueprint (render.yaml 자동 인식)
1. Render 대시보드 → **New → Blueprint**
2. 위 GitHub repo 선택 → `render.yaml`을 자동으로 읽어 설정 완료 → **Apply**

### 방법 2 — 수동 (Web Service)
1. Render → **New → Web Service** → GitHub repo 연결
2. 설정값:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Plan**: Free
   - **Health Check Path**: `/health`
3. (선택) Environment에 `USE_MOCK = false` 추가 (기본값도 false)
4. **Create Web Service** → 빌드 끝나면 `https://<이름>.onrender.com` 발급

## C. 확인
- `https://<이름>.onrender.com/health` → `{"ok": true, ...}` 면 서버 정상
- 루트 URL 접속 → 차트가 떠야 함

---

## 알아둘 점 (Render 무료 플랜)

- **콜드 스타트**: 15분 무접속 시 슬립 → 다음 접속 시 30~50초 기동 지연(정상).
- **Yahoo 차단 가능성**: 클라우드 IP라 Yahoo가 가끔 429로 막을 수 있습니다.
  그러면 자동으로 **Stooq로 폴백**합니다(일봉엔 충분). 그래도 데이터가 안 오면
  잠시 후 재시도하거나, 배포 자체가 잘 됐는지 확인하려면 Render 환경변수에서
  `USE_MOCK=true`로 잠깐 바꿔 합성 데이터로 화면을 확인하세요.
- **지연/일봉 데이터**라 볼린저·이평 기반 판단엔 영향 없습니다.

## 로컬에서도 돌리려면
```bash
pip install -r requirements.txt
cp .env.example .env
python app.py          # http://localhost:8000
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
| `app.py` | Flask 앱 (대시보드 + /chart, gunicorn 진입점 `app:app`) |
| `public_data.py` | 공개 시세 수집 (Yahoo → Stooq) |
| `indicators.py` | OHLCV → 차트 시리즈 + 의사결정 신호 |
| `dashboard.html` | 차트 UI + 의사결정 패널 |
| `render.yaml` | Render Blueprint 설정 |
| `requirements.txt` | 의존성 (gunicorn 포함) |

> 면허 있는 투자 자문이 아니며, 본 도구는 사용자의 매매 규칙을 정리·자동화하는 보조 장치입니다.
> 투자 판단과 결과의 책임은 전적으로 사용자에게 있습니다.

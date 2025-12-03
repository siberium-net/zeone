```
███████╗███████╗ ██████╗ ███╗   ██╗███████╗
╚══███╔╝██╔════╝██╔═══██╗████╗  ██║██╔════╝
  ███╔╝ █████╗  ██║   ██║██╔██╗ ██║█████╗  
 ███╔╝  ██╔══╝  ██║   ██║██║╚██╗██║██╔══╝  
███████╗███████╗╚██████╔╝██║ ╚████║███████╗
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
```

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.11-blue)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-black)](#)

# ZEONE — Sovereign Cognitive P2P Network

Автономная сеть для хранения, анализа и легальной обработки данных. Локальные узлы объединяют вычисления, DHT-хранилище, мульти-модальную аналитику (LLM + Vision), юридический фильтр и экономику доверия.

## 🚀 Abstract
ZEONE превращает любой узел в суверенный «когнитивный сервер», который умеет:
- находить и шифровать данные в распределённой сети без центра;
- анализировать текст, изображения, видео (Florence-2, InsightFace, OCR);
- автоматически проверять легальность (PII, комплаенс, юрисдикция);
- монетизировать ресурсы через Ledger, Trust Score и IOU.

## ✨ Key Features
- 🧠 **Cognitive Core** — Local LLM + Florence-2 Vision, OCR, InsightFace, pHash dedup.
- ⚖️ **Themis Module** — PII scanning (Presidio), AI-judge, geo rules, ZKP-ready.
- 👁️ **Neural Interface** — NiceGUI dashboard + Three.js 3D Neural Graph.
- 🌐 **Sovereign Network** — Asyncio transport, NaCl crypto, Kademlia DHT discovery/storage.
- 💱 **Economy Layer** — Ledger, Trust Score, IOU contracts, blocking for leechers.

## 🪙 Incentive Layer
- Узлы зарабатывают ZEO/USDT за:
  - GPU аренду (AI inference).
  - Хранение данных (DHT + persistence).
  - Пропускную способность (Relay/VPN).
- Гибрид расчётов:
  - **Fast Path (Off-chain IOU):** мгновенные микроплатежи в кредах через Ledger.
  - **Settlement Layer (On-chain):** периодические выплаты ERC-20 через `economy/chain.py` (Polygon/Arbitrum/Base).
- Поток ценности:
```
 Provider (GPU/Storage/Bandwidth)  <--- [ IOU Stream ] <---  Consumer
           |                                            |
           +------[ Batch Settlement / ERC-20 ]---------+
```

## 🧩 Architecture
- **Core:** Async transport (TCP/UDP), NaCl E2E, Kademlia DHT, BlockingTransport with Ledger accounting.
- **Cortex:** Archivist (text/OCR/vision/video), VisionEngine (Florence-2), FaceIndexer (InsightFace), MediaFingerprint (pHash).
- **Compliance:** PIIGuard (Presidio), ComplianceJudge (LLM), GeoRules, ZKP interfaces.
- **Economy:** Ledger (SQLite/async), Trust Score, IOU, blocking/balances on handshake.
- **UI:** NiceGUI WebUI + 3D NeuralVis; tabs for Peers, DHT, Cortex, Economy, Ingest.

## ⚡ Install & Start
Docker-first:
```bash
docker-compose up --build
```
Manual (Python 3.11):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # for Presidio
python main.py --port 9000 --webui --webui-port 8080
```
Connect a second node:
```bash
python main.py --port 9001 --bootstrap 127.0.0.1:9000 --webui --webui-port 8081
```

## 🛠 Roadmap
- [x] Async Core: transport, NaCl, Kademlia DHT
- [x] Ledger + Trust Score + IOU
- [x] Vision (Florence-2), OCR, pHash dedup, InsightFace clustering
- [x] Compliance: Presidio PII, AI judge, geo rules
- [x] WebUI + Neural Visualization
- [ ] ZKP proofs (Groth16) for age/identity
- [ ] Mobile/light clients

## 📜 License
Apache-2.0

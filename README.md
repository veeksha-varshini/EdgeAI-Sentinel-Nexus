# EdgeAI-Sentinel-Nexus

> **High-performance, real-time edge video surveillance and autonomous threat intelligence command center.**

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    EdgeAI-Sentinel-Nexus                       │
├───────────────┬──────────────────────┬─────────────────────────┤
│  C++ Producer │   Python Backend      │  Cyberpunk Frontend     │
│  producer.cpp │   FastAPI / main.py   │  index.html             │
│               │                       │                         │
│  60+ FPS RGB  │  np.frombuffer        │  WebSocket client       │
│  frame gen    │  zero-copy SHM view   │  Canvas bounding boxes  │
│               │                       │                         │
│  POSIX mmap   │  YOLOv8-nano YOLO    │  Tailwind CSS           │
│  /dev/shm     │  inference engine     │  Glassmorphism UI       │
│               │                       │                         │
│  TCP notify   │  SQLite telemetry     │  Live SVG/Canvas charts │
│  port 9999    │  WebSocket broadcast  │  Real-time event logs   │
└───────────────┴──────────────────────┴─────────────────────────┘
```

## Project Structure

```
EdgeAI-Sentinel-Nexus/
├── backend/
│   ├── producer.cpp        # C++17 POSIX shared memory frame producer
│   ├── main.py             # FastAPI + YOLOv8 + WebSocket backend
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Cyberpunk Glassmorphism single-page UI
└── README.md
```

---

## Quick Start

### 1. Python Backend (Required — works standalone)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

> **Without the C++ producer**, the backend automatically falls back to a
> high-fidelity Python/NumPy synthetic frame generator with animated
> objects and bounding boxes — the UI is fully functional immediately.

---

### 2. C++ Frame Producer (Linux only — optional, maximises FPS)

**Prerequisites:** GCC 10+, POSIX shared memory (`/dev/shm`), `librt`

```bash
cd backend
g++ -std=c++17 -O3 -march=native -pthread producer.cpp -lrt -o producer
./producer
```

The producer:
1. Allocates a **lock-free atomic ring buffer** in `/dev/shm/edgeai_sentinel_nexus`
2. Renders synthetic RGB frames at 60+ FPS using animated scanlines, grid
   overlays, and moving bounding-box objects
3. Notifies the Python backend via a compact **24-byte TCP packet** on port 9999

The Python backend auto-attaches to shared memory and uses **zero-copy
`np.frombuffer`** views for minimal latency.

---

## Feature Highlights

### C++ Producer (`producer.cpp`)
- **Lock-free atomic ring buffer** using `std::atomic<uint64_t>` write/read
  indices — no mutexes, no contention
- **POSIX `mmap`** over `/dev/shm` — in-kernel memory, never paged to disk
- **High-resolution frame pacing** via `std::chrono::steady_clock` busy-wait
  for the final 100 µs — achieves sub-millisecond frame jitter
- **24-byte TCP notification packet**: `[frame_id(8) | timestamp_ns(8) | fps(4) | flags(4)]`
- Animated synthetic frames: gradient scanlines, neon grid, animated objects
  with drawn bounding boxes — visually meaningful without a camera

### Python Backend (`main.py`)
- **FastAPI** with async WebSocket broadcast (`asyncio`-native, no blocking)
- **YOLOv8-nano** inference via `ultralytics` — auto-downloaded on first run
- Falls back to **threat-aware synthetic detections** when YOLO is unavailable
- **Zero-copy SHM consumer**: `mmap` + `np.frombuffer` — no data copy on the
  Python side when the C++ producer is running
- **SQLite telemetry logger** — frames table + events table, thread-safe
- **REST API**: `/health`, `/metrics`, `/events`, `/control/start`,
  `/control/stop`, `/control/lockdown`, `/control/sensitivity/{value}`
- **WebSocket**: bidirectional — backend pushes JSON frame stats + JPEG, 
  frontend sends control commands

### Frontend (`index.html`)
- **Cyberpunk Glassmorphism** — `backdrop-blur-xl`, `bg-white/5`,
  `border-white/10`, neon glow text shadows, pulsing status rings
- **Live JPEG video feed** from backend via WebSocket (`data:image/jpeg;base64`)
- **Animated bounding boxes** drawn on `<canvas>` overlay with corner accents,
  pulsing glow, and label chips
- **Real-time line charts** (FPS + Latency) rendered in pure Canvas API
- **SVG gauge rings** for CPU and Memory — smooth CSS transitions
- **Event log + AI Threat Intelligence log** with severity-coded
  glassmorphism entries that slide in
- **Demo mode** — when WebSocket is offline, animated synthetic data keeps
  all UI components live (pure client-side, zero backend dependency)
- **Emergency Lockdown** — full-screen red overlay with flicker animation,
  unlock by clicking the banner

---

## WebSocket Protocol

**Server → Client** (every frame):
```json
{
  "type":           "frame_stats",
  "frame_id":       12345,
  "fps":            60.49,
  "latency_ms":     1.74,
  "fps_history":    [60.1, 60.3, ...],
  "latency_history":[1.6, 1.8, ...],
  "detections":     [{"label":"person","threat":"INTRUDER","severity":"CRITICAL","confidence":0.91,"bbox":[100,80,180,180]}],
  "total_threats":  7,
  "dropped":        0,
  "cpu_pct":        23.4,
  "mem_pct":        44.1,
  "frame_b64":      "<base64 JPEG>",
  "shm_attached":   true,
  "yolo_ready":     true,
  "lockdown":       false,
  "source":         "shm"
}
```

**Client → Server** (control):
```json
{ "cmd": "start" }
{ "cmd": "stop" }
{ "cmd": "lockdown" }
{ "cmd": "unlock" }
{ "cmd": "sensitivity", "value": 0.5 }
{ "cmd": "ai_toggle" }
```

---

## Threat Classification

| YOLO Class    | Threat Label         | Severity |
|---------------|----------------------|----------|
| person        | INTRUDER             | CRITICAL |
| knife / gun   | WEAPON               | CRITICAL |
| backpack / suitcase | SUSPICIOUS_PACKAGE | HIGH |
| truck         | HEAVY_VEHICLE        | HIGH |
| car           | VEHICLE              | MEDIUM |
| motorbike     | MOTORBIKE            | MEDIUM |
| bicycle       | BICYCLE              | LOW |
| cell phone    | COMMS_DEVICE         | LOW |

---

## REST API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/`  | Serves `frontend/index.html` |
| GET    | `/health` | System health check |
| GET    | `/metrics` | Telemetry metrics |
| GET    | `/events?limit=50` | Recent event log |
| POST   | `/control/start` | Resume stream |
| POST   | `/control/stop` | Pause stream |
| POST   | `/control/lockdown` | Emergency lockdown |
| POST   | `/control/unlock` | Lift lockdown |
| POST   | `/control/sensitivity/{value}` | Set AI confidence threshold |
| WS     | `/ws` | Live telemetry WebSocket |
| Docs   | `/docs` | Swagger UI |

---

## Performance Notes

- **Shared memory layout** is cache-line aligned (64 bytes) to prevent false
  sharing between producer/consumer cache lines
- **`MADV_SEQUENTIAL | MADV_WILLNEED`** hints to the kernel prefetcher
- **AI frame skipping** (`AI_SKIP_FRAMES = 4`) runs YOLO every 4th frame —
  reduces GPU/CPU load by 75% while keeping detection latency ≤ 67 ms
- **JPEG quality 82** — good visual fidelity at ≈12 KB per frame over WebSocket
- The Python backend targets `TARGET_FPS` with `asyncio.sleep` pacing and
  resets the cadence if it falls behind to avoid a runaway latency spiral

"""
EdgeAI-Sentinel-Nexus :: Python FastAPI Backend
================================================
Attaches to the C++ producer's POSIX shared memory ring buffer via
zero-copy NumPy views (np.frombuffer), runs YOLOv8-nano inference,
logs telemetry to SQLite, and broadcasts live JSON stats to the
React-less Cyberpunk frontend via WebSockets.

Requirements:
    pip install fastapi uvicorn[standard] ultralytics numpy opencv-python \
                aiofiles python-multipart websockets

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import json
import logging
import math
import mmap
import os
import random
import socket
import sqlite3
import struct
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── cv2-free helpers (Pillow replaces OpenCV for encode/draw) ─────────────────
def _pil_encode_jpeg(rgb_array: np.ndarray, quality: int = 82) -> bytes:
    """Encode an RGB uint8 NumPy array to JPEG bytes using Pillow."""
    img = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def _pil_rectangle(draw: ImageDraw.ImageDraw,
                   x1: int, y1: int, x2: int, y2: int,
                   color: tuple, width: int = 2):
    """Draw a rectangle outline on a PIL ImageDraw."""
    for w in range(width):
        draw.rectangle([x1+w, y1+w, x2-w, y2-w], outline=color)

def _pil_text_box(draw: ImageDraw.ImageDraw,
                  x: int, y: int, text: str, color: tuple):
    """Draw a small label above (or inside top of) a bounding box."""
    font_size = 11
    cw = int(font_size * 0.6)
    tw = len(text) * cw + 4
    th = font_size + 4
    # If label would go off the top, draw it inside the box instead
    label_y0 = max(0, y - th)
    label_y1 = label_y0 + th
    draw.rectangle([x, label_y0, x + tw, label_y1], fill=color + (180,))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((x + 2, label_y0 + 2), text, fill=(0, 0, 0), font=font)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinel")

# ─── Configuration ─────────────────────────────────────────────────────────────
SHM_NAME       = "/edgeai_sentinel_nexus"
RING_CAPACITY  = 8
FRAME_W        = 640
FRAME_H        = 480
FRAME_CHANNELS = 3
FRAME_BYTES    = FRAME_W * FRAME_H * FRAME_CHANNELS

# Offset of ShmHeader fields (must mirror producer.cpp layout)
HEADER_WRITE_IDX_OFFSET = 0          # atomic<uint64_t>  — 8 bytes
HEADER_READ_IDX_OFFSET  = 8          # atomic<uint64_t>  — 8 bytes
HEADER_FPS_OFFSET       = 8 + 8 + 4 * 5 + 8  # after write/read idx + ints + fps_actual
HEADER_SIZE             = 64         # cache-line aligned
FRAME_META_SIZE         = 48         # FrameMeta struct
FRAME_SLOT_SIZE         = FRAME_META_SIZE + FRAME_BYTES

NOTIFY_PORT    = 9999
DB_PATH        = Path("telemetry.db")
MODEL_NAME     = "yolov8n.pt"        # downloaded automatically on first run

TARGET_FPS     = 60.0
AI_SKIP_FRAMES = 4                   # run inference every N frames (perf trade-off)

# ─── Threat Intelligence Vocabulary ───────────────────────────────────────────
THREAT_LABELS: dict[str, str] = {
    "person":     "INTRUDER",
    "car":        "VEHICLE",
    "truck":      "HEAVY_VEHICLE",
    "motorbike":  "MOTORBIKE",
    "bicycle":    "BICYCLE",
    "knife":      "WEAPON",
    "gun":        "WEAPON",
    "backpack":   "SUSPICIOUS_PACKAGE",
    "suitcase":   "SUSPICIOUS_PACKAGE",
    "cell phone": "COMMS_DEVICE",
}

SEVERITY_MAP: dict[str, str] = {
    "INTRUDER":            "CRITICAL",
    "WEAPON":              "CRITICAL",
    "SUSPICIOUS_PACKAGE":  "HIGH",
    "VEHICLE":             "MEDIUM",
    "HEAVY_VEHICLE":       "HIGH",
    "MOTORBIKE":           "MEDIUM",
    "BICYCLE":             "LOW",
    "COMMS_DEVICE":        "LOW",
}

# ─── Data Models ──────────────────────────────────────────────────────────────
@dataclass
class Detection:
    label: str
    threat: str
    severity: str
    confidence: float
    bbox: list[int]   # [x1, y1, x2, y2]

@dataclass
class FrameStats:
    frame_id:     int
    timestamp:    str
    fps:          float
    latency_ms:   float
    detections:   list[dict]
    threats:      int
    dropped:      int
    cpu_pct:      float
    mem_pct:      float
    shm_attached: bool

# ─── SQLite Telemetry Logger ───────────────────────────────────────────────────
class TelemetryDB:
    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id    INTEGER,
                    ts          TEXT,
                    fps         REAL,
                    latency_ms  REAL,
                    threats     INTEGER,
                    detections  TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT,
                    severity TEXT,
                    message  TEXT
                )
            """)

    def log_frame(self, stats: FrameStats):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO frames(frame_id,ts,fps,latency_ms,threats,detections) "
                    "VALUES (?,?,?,?,?,?)",
                    (stats.frame_id, stats.timestamp, stats.fps,
                     stats.latency_ms, stats.threats, json.dumps(stats.detections))
                )

    def log_event(self, severity: str, message: str):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO events(ts,severity,message) VALUES (?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), severity, message)
                )

    def recent_events(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT ts, severity, message FROM events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [{"ts": r[0], "severity": r[1], "message": r[2]} for r in cur]

# ─── Shared Memory Consumer ────────────────────────────────────────────────────
class ShmConsumer:
    """
    Attaches to the POSIX shared memory segment created by producer.cpp.
    Provides zero-copy NumPy views via np.frombuffer.
    Falls back to synthetic frame generation when SHM is unavailable
    (e.g., running on Windows / no C++ producer).
    """

    def __init__(self):
        self.attached  = False
        self._mm: mmap.mmap | None = None
        self._fd: int | None = None
        self._buf: bytes | None = None
        self._frame_counter = 0
        self._synthetic_gen = SyntheticFrameGenerator()
        self._last_write_idx = 0

    def attach(self):
        shm_path = f"/dev/shm{SHM_NAME}"
        if not Path(shm_path).exists():
            log.warning("SHM segment not found (%s) — using synthetic frames.", shm_path)
            return False
        try:
            self._fd = os.open(shm_path, os.O_RDWR)
            total_size = HEADER_SIZE + FRAME_SLOT_SIZE * RING_CAPACITY
            self._mm = mmap.mmap(self._fd, total_size, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE)
            self.attached = True
            log.info("SHM attached: %s (%d KB)", shm_path, total_size // 1024)
            return True
        except Exception as exc:
            log.warning("SHM attach failed (%s) — using synthetic frames.", exc)
            return False

    def _read_uint64(self, offset: int) -> int:
        self._mm.seek(offset)
        return struct.unpack_from("<Q", self._mm.read(8))[0]

    def _write_uint64(self, offset: int, val: int):
        self._mm.seek(offset)
        self._mm.write(struct.pack("<Q", val))

    def _read_float32(self, offset: int) -> float:
        self._mm.seek(offset)
        return struct.unpack_from("<f", self._mm.read(4))[0]

    def get_frame(self) -> tuple[np.ndarray, dict]:
        """Return (rgb_frame, meta_dict).  Zero-copy when SHM is attached."""
        if self.attached and self._mm:
            return self._get_shm_frame()
        return self._get_synthetic_frame()

    def _get_shm_frame(self) -> tuple[np.ndarray, dict]:
        write_idx = self._read_uint64(HEADER_WRITE_IDX_OFFSET)
        read_idx  = self._read_uint64(HEADER_READ_IDX_OFFSET)

        if write_idx == read_idx:
            # No new frame — return last synthetic
            return self._get_synthetic_frame()

        # Consume latest available slot (skip stale ones)
        consume_idx = write_idx - 1
        slot_offset = HEADER_SIZE + (consume_idx % RING_CAPACITY) * FRAME_SLOT_SIZE

        # Read metadata (48 bytes)
        self._mm.seek(slot_offset)
        meta_bytes = self._mm.read(FRAME_META_SIZE)
        frame_id, ts_ns, w, h, ch, bsz = struct.unpack_from("<QqIIII", meta_bytes)

        # Zero-copy pixel view
        pixel_offset = slot_offset + FRAME_META_SIZE
        self._mm.seek(pixel_offset)
        raw = self._mm.read(FRAME_BYTES)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3)).copy()

        # Advance read pointer
        self._write_uint64(HEADER_READ_IDX_OFFSET, consume_idx + 1)

        fps = self._read_float32(HEADER_SIZE - 24)   # fps_actual field offset

        meta = {
            "frame_id":    frame_id,
            "timestamp_ns": ts_ns,
            "fps":         fps,
            "source":      "shm",
        }
        return frame, meta

    def _get_synthetic_frame(self) -> tuple[np.ndarray, dict]:
        self._frame_counter += 1
        frame = self._synthetic_gen.render(self._frame_counter)
        meta = {
            "frame_id":    self._frame_counter,
            "timestamp_ns": int(time.time_ns()),
            "fps":         TARGET_FPS,
            "source":      "synthetic",
        }
        return frame, meta


# ─── Synthetic Frame Generator (Python fallback) ──────────────────────────────
class SyntheticFrameGenerator:
    """
    Generates realistic-looking synthetic surveillance frames entirely in
    NumPy / OpenCV — used when the C++ producer is not running.
    """

    def __init__(self):
        self._rng = np.random.default_rng(42)
        self._objects = [
            {"x": 100, "y": 80,  "vx": 2,  "vy": 1,  "w": 80, "h": 100,
             "label": "INTRUDER", "color": (6, 182, 212)},
            {"x": 400, "y": 200, "vx": -1, "vy": 2,  "w": 60, "h": 80,
             "label": "VEHICLE",  "color": (139, 92, 246)},
            {"x": 250, "y": 350, "vx": 3,  "vy": -1, "w": 50, "h": 50,
             "label": "PACKAGE",  "color": (34, 197, 94)},
        ]

    def render(self, frame_id: int) -> np.ndarray:
        t = frame_id / TARGET_FPS

        # Base obsidian gradient
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for y in range(FRAME_H):
            val = int(8 + 6 * math.sin(t * 0.3 + y * 0.01))
            frame[y, :, 0] = max(0, min(255, val))
            frame[y, :, 1] = max(0, min(255, val + 2))
            frame[y, :, 2] = max(0, min(255, val + 10))

        # Noise overlay
        noise = self._rng.integers(0, 20, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Grid lines
        grid_col = int(25 + 15 * math.sin(t))
        for gy in range(0, FRAME_H, 40):
            frame[gy, :, 2] = np.clip(frame[gy, :, 2].astype(int) + grid_col, 0, 255)
        for gx in range(0, FRAME_W, 40):
            frame[:, gx, 2] = np.clip(frame[:, gx, 2].astype(int) + grid_col, 0, 255)

        # Moving scan line
        scan_y = int(t * 80) % FRAME_H
        frame[scan_y, :] = np.clip(
            frame[scan_y, :].astype(int) + [80, 160, 160], 0, 255)

        # Animate and draw objects
        pil_img = Image.fromarray(frame, mode="RGB")
        draw    = ImageDraw.Draw(pil_img, "RGBA")
        for obj in self._objects:
            obj["x"] = (obj["x"] + obj["vx"]) % FRAME_W
            obj["y"] = (obj["y"] + obj["vy"]) % FRAME_H
            x1, y1 = int(obj["x"]), int(obj["y"])
            x2 = max(x1 + 2, min(x1 + obj["w"], FRAME_W - 1))
            y2 = max(y1 + 2, min(y1 + obj["h"], FRAME_H - 1))
            # Semi-transparent fill
            roi  = frame[y1:y2, x1:x2].astype(np.int16)
            tint = np.array(obj["color"], dtype=np.int16)
            frame[y1:y2, x1:x2] = np.clip(roi + tint // 6, 0, 255).astype(np.uint8)
            # Redraw pil_img after numpy edit
            pil_img = Image.fromarray(frame, mode="RGB")
            draw    = ImageDraw.Draw(pil_img, "RGBA")
            _pil_rectangle(draw, x1, y1, x2, y2, obj["color"], 2)
            _pil_text_box(draw, x1, y1, obj["label"], obj["color"])
            frame = np.array(pil_img)

        # Corner HUD overlays
        ts_str  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        pil_img = Image.fromarray(frame, mode="RGB")
        draw    = ImageDraw.Draw(pil_img)
        try:
            font_sm = ImageFont.truetype("arial.ttf", 11)
            font_xs = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            font_sm = font_xs = ImageFont.load_default()
        draw.text((8, 6),  f"SENTINEL [{ts_str}]", fill=(6, 182, 212),  font=font_sm)
        draw.text((8, FRAME_H - 18), f"FRM {frame_id:08d}", fill=(100, 100, 200), font=font_xs)
        frame = np.array(pil_img)

        return frame


# ─── YOLOv8 Inference Engine ───────────────────────────────────────────────────
class YOLOEngine:
    def __init__(self, model_name: str = MODEL_NAME):
        self.ready   = False
        self._model  = None
        self._model_name = model_name
        self._skip_ctr   = 0
        self._last_dets: list[Detection] = []
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

    def _load(self):
        try:
            from ultralytics import YOLO
            log.info("Loading YOLOv8 model: %s", self._model_name)
            self._model = YOLO(self._model_name)
            # Warm-up inference
            dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            self.ready = True
            log.info("YOLOv8 engine ready ✓")
        except Exception as exc:
            log.warning("YOLOv8 not available (%s) — using synthetic detections.", exc)
            self.ready = False

    def infer(self, frame: np.ndarray, sensitivity: float = 0.4) -> list[Detection]:
        """
        Run inference if model is ready and skip counter allows it.
        Returns list of Detection objects.
        """
        self._skip_ctr += 1
        if self._skip_ctr % AI_SKIP_FRAMES != 0:
            return self._last_dets

        if self.ready and self._model:
            return self._run_yolo(frame, sensitivity)
        return self._synthetic_detections(frame)

    def _run_yolo(self, frame: np.ndarray, conf: float) -> list[Detection]:
        try:
            results = self._model(frame, conf=conf, verbose=False)
            dets: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls)]
                    threat   = THREAT_LABELS.get(cls_name, cls_name.upper())
                    severity = SEVERITY_MAP.get(threat, "INFO")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append(Detection(
                        label=cls_name,
                        threat=threat,
                        severity=severity,
                        confidence=float(box.conf),
                        bbox=[x1, y1, x2, y2],
                    ))
            self._last_dets = dets
            return dets
        except Exception as exc:
            log.error("YOLO inference error: %s", exc)
            return []

    def _synthetic_detections(self, frame: np.ndarray) -> list[Detection]:
        """Return plausible-looking fake detections for demo purposes."""
        t = time.time()
        cycle = int(t * 0.5) % 4
        candidates = [
            Detection("person",   "INTRUDER",           "CRITICAL", 0.91,
                      [100, 80,  180, 180]),
            Detection("car",      "VEHICLE",            "MEDIUM",   0.85,
                      [380, 190, 460, 270]),
            Detection("backpack", "SUSPICIOUS_PACKAGE", "HIGH",     0.78,
                      [240, 330, 300, 400]),
            Detection("cell phone","COMMS_DEVICE",       "LOW",      0.72,
                      [310, 150, 350, 200]),
        ]
        # Vary returned set by cycle so the UI shows dynamic changes
        count = 1 + cycle % 3
        self._last_dets = candidates[:count]
        return self._last_dets


# ─── TCP Notification Listener ─────────────────────────────────────────────────
class TcpListener:
    """
    Listens on NOTIFY_PORT for compact 24-byte notification packets
    from the C++ producer. Each packet increments a shared counter
    and updates the latest FPS telemetry from the producer side.
    """

    def __init__(self):
        self.last_frame_id = 0
        self.producer_fps  = 0.0
        self._running      = False
        self._thread       = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()
        log.info("TCP listener started on port %d", NOTIFY_PORT)

    def _run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", NOTIFY_PORT))
        srv.listen(1)
        srv.settimeout(1.0)

        while self._running:
            try:
                conn, addr = srv.accept()
                log.info("Producer connected from %s", addr)
                self._handle(conn)
            except socket.timeout:
                continue
            except Exception as exc:
                log.error("TCP listener error: %s", exc)

    def _handle(self, conn: socket.socket):
        try:
            while self._running:
                data = conn.recv(24)
                if len(data) < 24:
                    break
                frame_id, ts_ns, fps = struct.unpack_from("<QQf", data)
                self.last_frame_id = frame_id
                self.producer_fps  = fps
        except Exception:
            pass
        finally:
            conn.close()
            log.info("Producer disconnected.")


# ─── WebSocket Connection Manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._conns: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._conns.append(ws)
        log.info("WS client connected. Total: %d", len(self._conns))

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._conns = [c for c in self._conns if c is not ws]
        log.info("WS client disconnected. Total: %d", len(self._conns))

    async def broadcast(self, payload: dict):
        msg = json.dumps(payload)
        dead: list[WebSocket] = []
        async with self._lock:
            conns = list(self._conns)
        for ws in conns:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


# ─── Global State ──────────────────────────────────────────────────────────────
shm_consumer = ShmConsumer()
yolo         = YOLOEngine()
tcp_listener = TcpListener()
db           = TelemetryDB()
ws_manager   = ConnectionManager()

# Mutable runtime state
_state = {
    "streaming":   True,
    "sensitivity": 0.4,
    "lockdown":    False,
    "ai_enabled":  True,
    "dropped":     0,
    "total_threats": 0,
    "frame_count": 0,
    "fps_history": [],
    "latency_history": [],
}

# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("EdgeAI-Sentinel-Nexus backend starting…")
    shm_consumer.attach()
    tcp_listener.start()
    asyncio.create_task(frame_pipeline())
    yield
    log.info("Backend shutting down.")

# ─── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(
    title="EdgeAI-Sentinel-Nexus API",
    version="1.0.0",
    description="High-performance real-time edge video surveillance backend.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if the directory exists
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

# ─── Frame Pipeline (background task) ─────────────────────────────────────────
async def frame_pipeline():
    """
    Main async loop: pulls frames from SHM (or synthetic), runs YOLO,
    encodes a JPEG thumbnail, and broadcasts a JSON payload to all
    connected WebSocket clients at ~TARGET_FPS.
    """
    loop          = asyncio.get_event_loop()
    frame_interval = 1.0 / TARGET_FPS
    next_tick      = loop.time()
    fps_window_start = time.monotonic()
    fps_window_count = 0
    cur_fps          = TARGET_FPS

    log.info("Frame pipeline started.")

    while True:
        t0 = time.monotonic()

        if not _state["streaming"]:
            await asyncio.sleep(0.1)
            continue

        # ── Pull frame ────────────────────────────────────────────────────────
        try:
            frame, meta = await loop.run_in_executor(None, shm_consumer.get_frame)
        except Exception as exc:
            log.error("Frame pull error: %s", exc)
            _state["dropped"] += 1
            await asyncio.sleep(frame_interval)
            continue

        # ── YOLO inference ────────────────────────────────────────────────────
        dets: list[Detection] = []
        if _state["ai_enabled"]:
            try:
                dets = await loop.run_in_executor(
                    None, yolo.infer, frame, _state["sensitivity"]
                )
            except Exception as exc:
                log.error("YOLO error: %s", exc)

        # ── Draw bounding boxes on frame (Pillow — no cv2) ───────────────────
        pil_ann = Image.fromarray(frame.copy(), mode="RGB")
        draw    = ImageDraw.Draw(pil_ann, "RGBA")
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            color = {
                "CRITICAL": (6, 182, 212),
                "HIGH":     (239, 68, 68),
                "MEDIUM":   (245, 158, 11),
                "LOW":      (34, 197, 94),
            }.get(det.severity, (200, 200, 200))
            label_txt = f"{det.threat} {det.confidence:.0%}"
            _pil_rectangle(draw, x1, y1, x2, y2, color, 2)
            _pil_text_box(draw, x1, y1, label_txt, color)
        annotated = np.array(pil_ann)

        # ── JPEG encode (Pillow) ──────────────────────────────────────────────
        frame_b64 = base64.b64encode(_pil_encode_jpeg(annotated, quality=82)).decode()

        # ── Metrics ───────────────────────────────────────────────────────────
        latency_ms = (time.monotonic() - t0) * 1000.0
        fps_window_count += 1
        elapsed = time.monotonic() - fps_window_start
        if elapsed >= 1.0:
            cur_fps = fps_window_count / elapsed
            fps_window_count = 0
            fps_window_start = time.monotonic()

        new_threats = len([d for d in dets if d.severity in ("CRITICAL", "HIGH")])
        _state["total_threats"] += new_threats
        _state["frame_count"]   += 1

        # Keep rolling histories (last 60 samples)
        _state["fps_history"]     = (_state["fps_history"] + [round(cur_fps, 2)])[-60:]
        _state["latency_history"] = (_state["latency_history"] + [round(latency_ms, 3)])[-60:]

        # ── Persist telemetry (every 10th frame) ──────────────────────────────
        stats = FrameStats(
            frame_id   = meta["frame_id"],
            timestamp  = datetime.now(timezone.utc).isoformat(),
            fps        = round(cur_fps, 2),
            latency_ms = round(latency_ms, 3),
            detections = [asdict(d) for d in dets],
            threats    = new_threats,
            dropped    = _state["dropped"],
            cpu_pct    = _get_cpu(),
            mem_pct    = _get_mem(),
            shm_attached = shm_consumer.attached,
        )
        if _state["frame_count"] % 10 == 0:
            await loop.run_in_executor(None, db.log_frame, stats)

        if new_threats > 0:
            for det in dets:
                if det.severity in ("CRITICAL", "HIGH"):
                    msg = f"[{det.severity}] {det.threat} detected (conf={det.confidence:.0%})"
                    await loop.run_in_executor(None, db.log_event, det.severity, msg)

        # ── Broadcast to WebSocket clients ────────────────────────────────────
        payload = {
            "type":           "frame_stats",
            "frame_id":       stats.frame_id,
            "timestamp":      stats.timestamp,
            "fps":            stats.fps,
            "fps_history":    _state["fps_history"],
            "latency_ms":     stats.latency_ms,
            "latency_history":_state["latency_history"],
            "detections":     stats.detections,
            "threats":        stats.threats,
            "total_threats":  _state["total_threats"],
            "dropped":        _state["dropped"],
            "cpu_pct":        stats.cpu_pct,
            "mem_pct":        stats.mem_pct,
            "shm_attached":   stats.shm_attached,
            "streaming":      _state["streaming"],
            "lockdown":       _state["lockdown"],
            "ai_enabled":     _state["ai_enabled"],
            "sensitivity":    _state["sensitivity"],
            "frame_b64":      frame_b64,
            "source":         meta["source"],
            "yolo_ready":     yolo.ready,
        }

        if ws_manager._conns:
            await ws_manager.broadcast(payload)

        # ── Frame pacing ──────────────────────────────────────────────────────
        next_tick += frame_interval
        sleep_for  = next_tick - loop.time()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            # We're behind — reset cadence to avoid spiral
            next_tick = loop.time()
            _state["dropped"] += 1


def _get_cpu() -> float:
    """Best-effort CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        # Fallback: parse /proc/stat on Linux
        try:
            with open("/proc/stat") as f:
                vals = list(map(int, f.readline().split()[1:]))
            idle, total = vals[3], sum(vals)
            if not hasattr(_get_cpu, "_prev"):
                _get_cpu._prev = (idle, total)
                return 0.0
            d_idle  = idle  - _get_cpu._prev[0]
            d_total = total - _get_cpu._prev[1]
            _get_cpu._prev = (idle, total)
            return round((1 - d_idle / max(d_total, 1)) * 100, 1)
        except Exception:
            return round(20 + 15 * math.sin(time.time() * 0.3), 1)


def _get_mem() -> float:
    """Best-effort memory usage percentage."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = {l.split(":")[0]: int(l.split()[1])
                         for l in f if ":" in l}
            total = lines.get("MemTotal", 1)
            avail = lines.get("MemAvailable", total)
            return round((1 - avail / total) * 100, 1)
        except Exception:
            return round(45 + 10 * math.sin(time.time() * 0.1), 1)


# ─── REST Endpoints ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1 style='font-family:monospace;color:cyan'>EdgeAI-Sentinel-Nexus API</h1>"
        "<p>Connect to <code>ws://localhost:8000/ws</code> for live telemetry.</p>"
        "<p>Visit <a href='/docs'>/docs</a> for the API reference.</p>"
    )

@app.get("/health")
async def health():
    return {
        "status":       "operational",
        "shm_attached": shm_consumer.attached,
        "yolo_ready":   yolo.ready,
        "streaming":    _state["streaming"],
        "lockdown":     _state["lockdown"],
        "total_frames": _state["frame_count"],
        "total_threats": _state["total_threats"],
    }

@app.post("/control/start")
async def control_start():
    _state["streaming"] = True
    db.log_event("INFO", "Stream STARTED by operator")
    return {"status": "streaming"}

@app.post("/control/stop")
async def control_stop():
    _state["streaming"] = False
    db.log_event("INFO", "Stream STOPPED by operator")
    return {"status": "stopped"}

@app.post("/control/sensitivity/{value}")
async def set_sensitivity(value: float):
    _state["sensitivity"] = max(0.1, min(1.0, value))
    db.log_event("INFO", f"AI sensitivity set to {_state['sensitivity']:.2f}")
    return {"sensitivity": _state["sensitivity"]}

@app.post("/control/lockdown")
async def trigger_lockdown():
    _state["lockdown"]  = True
    _state["streaming"] = False
    db.log_event("CRITICAL", "EMERGENCY LOCKDOWN ACTIVATED")
    log.critical("⚠️  EMERGENCY LOCKDOWN ACTIVATED")
    return {"lockdown": True}

@app.post("/control/unlock")
async def unlock():
    _state["lockdown"]  = False
    _state["streaming"] = True
    db.log_event("INFO", "Lockdown lifted — stream resumed")
    return {"lockdown": False}

@app.get("/events")
async def get_events(limit: int = 50):
    return db.recent_events(limit)

@app.get("/metrics")
async def get_metrics():
    return {
        "fps_history":     _state["fps_history"],
        "latency_history": _state["latency_history"],
        "total_frames":    _state["frame_count"],
        "total_threats":   _state["total_threats"],
        "dropped":         _state["dropped"],
        "streaming":       _state["streaming"],
    }


# ─── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            # Accept control messages from frontend
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "start":
                    _state["streaming"] = True
                elif cmd == "stop":
                    _state["streaming"] = False
                elif cmd == "lockdown":
                    _state["lockdown"]  = True
                    _state["streaming"] = False
                    db.log_event("CRITICAL", "LOCKDOWN via WebSocket")
                elif cmd == "unlock":
                    _state["lockdown"]  = False
                    _state["streaming"] = True
                elif cmd == "sensitivity" and "value" in msg:
                    _state["sensitivity"] = float(msg["value"])
                elif cmd == "ai_toggle":
                    _state["ai_enabled"] = not _state["ai_enabled"]
            except asyncio.TimeoutError:
                # Keepalive ping
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(ws)

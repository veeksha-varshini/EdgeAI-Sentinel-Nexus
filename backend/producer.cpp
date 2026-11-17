/**
 * EdgeAI-Sentinel-Nexus :: Frame Producer
 * =========================================
 * C++17 high-frequency video frame generator.
 * Allocates a lock-free atomic ring buffer in POSIX shared memory (/dev/shm),
 * streams synthetic RGB frames at 60+ FPS, and notifies the Python backend
 * over a TCP socket.
 *
 * Build (Linux):
 *   g++ -std=c++17 -O3 -march=native -pthread producer.cpp -lrt -o producer
 *
 * Run:
 *   ./producer
 */

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// POSIX / Linux headers
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

// ─── Configuration ────────────────────────────────────────────────────────────
static constexpr const char* SHM_NAME      = "/edgeai_sentinel_nexus";
static constexpr int         RING_CAPACITY  = 8;          // number of frame slots
static constexpr int         FRAME_W        = 640;
static constexpr int         FRAME_H        = 480;
static constexpr int         FRAME_CHANNELS = 3;          // RGB
static constexpr int         FRAME_BYTES    = FRAME_W * FRAME_H * FRAME_CHANNELS;
static constexpr double      TARGET_FPS     = 60.0;
static constexpr uint16_t    NOTIFY_PORT    = 9999;       // TCP notification port
static constexpr const char* NOTIFY_HOST    = "127.0.0.1";

// ─── Shared Memory Layout ─────────────────────────────────────────────────────
//
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │  ShmHeader  (64-byte cache-line aligned)                             │
//  │    write_idx : atomic<uint64_t>  – producer advances this           │
//  │    read_idx  : atomic<uint64_t>  – consumer advances this           │
//  │    frame_w / frame_h / channels / frame_bytes                       │
//  │    fps_actual : atomic<float>    – live FPS telemetry               │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  FrameSlot[RING_CAPACITY]                                           │
//  │    each slot = { FrameMeta, uint8_t pixels[FRAME_BYTES] }          │
//  └──────────────────────────────────────────────────────────────────────┘

struct alignas(64) ShmHeader {
    std::atomic<uint64_t> write_idx{0};  // monotonic write counter
    std::atomic<uint64_t> read_idx{0};   // monotonic read counter (set by consumer)
    int32_t  frame_w{FRAME_W};
    int32_t  frame_h{FRAME_H};
    int32_t  channels{FRAME_CHANNELS};
    int32_t  frame_bytes{FRAME_BYTES};
    int32_t  ring_capacity{RING_CAPACITY};
    std::atomic<float> fps_actual{0.0f};
    std::atomic<uint64_t> total_frames{0};
    std::atomic<uint32_t> producer_pid{0};
    uint8_t  _pad[64 - sizeof(std::atomic<uint64_t>) * 3
                      - sizeof(std::atomic<float>)
                      - sizeof(int32_t) * 5
                      - sizeof(std::atomic<uint32_t>)]{};
};

struct FrameMeta {
    uint64_t frame_id;
    int64_t  timestamp_ns;   // nanoseconds since epoch
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t byte_size;
    uint8_t  _pad[8]{};
};

struct FrameSlot {
    FrameMeta meta;
    uint8_t   pixels[FRAME_BYTES];
};

static constexpr size_t SHM_SIZE = sizeof(ShmHeader) + sizeof(FrameSlot) * RING_CAPACITY;

// ─── Synthetic Frame Generator ────────────────────────────────────────────────
class FrameGenerator {
public:
    FrameGenerator()
        : rng_(std::random_device{}()),
          dist_noise_(0, 30),
          dist_obj_(0, FRAME_W - 80)
    {}

    void render(uint8_t* buf, uint64_t frame_id) {
        const double t = frame_id / TARGET_FPS;

        // Scanline gradient – deep obsidian base
        for (int y = 0; y < FRAME_H; ++y) {
            uint8_t base_r = static_cast<uint8_t>(8  + 4  * std::sin(t * 0.3 + y * 0.01));
            uint8_t base_g = static_cast<uint8_t>(10 + 6  * std::cos(t * 0.2 + y * 0.015));
            uint8_t base_b = static_cast<uint8_t>(18 + 12 * std::sin(t * 0.4 + y * 0.008));

            for (int x = 0; x < FRAME_W; ++x) {
                int idx = (y * FRAME_W + x) * 3;
                buf[idx + 0] = static_cast<uint8_t>(
                    std::clamp(base_r + dist_noise_(rng_), 0, 255));
                buf[idx + 1] = static_cast<uint8_t>(
                    std::clamp(base_g + dist_noise_(rng_), 0, 255));
                buf[idx + 2] = static_cast<uint8_t>(
                    std::clamp(base_b + dist_noise_(rng_), 0, 255));
            }
        }

        // Draw animated neon grid lines
        draw_grid(buf, t);

        // Draw 1-3 simulated objects (bright bounding-box region fills)
        int n_objs = 1 + (frame_id % 3);
        for (int i = 0; i < n_objs; ++i) {
            draw_object(buf, frame_id, i, t);
        }

        // Draw moving scan line
        int scan_y = static_cast<int>((t * 80.0)) % FRAME_H;
        draw_scanline(buf, scan_y);
    }

private:
    std::mt19937 rng_;
    std::uniform_int_distribution<int> dist_noise_;
    std::uniform_int_distribution<int> dist_obj_;

    void draw_grid(uint8_t* buf, double t) {
        uint8_t gcol = static_cast<uint8_t>(30 + 20 * std::sin(t));
        for (int y = 0; y < FRAME_H; y += 40) {
            for (int x = 0; x < FRAME_W; ++x) {
                int idx = (y * FRAME_W + x) * 3;
                buf[idx + 2] = std::min(255, (int)buf[idx + 2] + gcol);
            }
        }
        for (int x = 0; x < FRAME_W; x += 40) {
            for (int y = 0; y < FRAME_H; ++y) {
                int idx = (y * FRAME_W + x) * 3;
                buf[idx + 2] = std::min(255, (int)buf[idx + 2] + gcol);
            }
        }
    }

    void draw_object(uint8_t* buf, uint64_t fid, int obj_id, double t) {
        // Each object has a pseudo-random trajectory
        int seed_x = (int)(fid / 30 + obj_id * 137) % (FRAME_W - 80);
        int seed_y = (int)(fid / 45 + obj_id * 97)  % (FRAME_H - 80);
        int x0 = std::abs(seed_x);
        int y0 = std::abs(seed_y);
        int bw = 60 + obj_id * 20;
        int bh = 50 + obj_id * 15;

        // Fill interior with semi-bright pixels
        uint8_t fill_r = (obj_id == 0) ? 6  : (obj_id == 1) ? 139 : 34;
        uint8_t fill_g = (obj_id == 0) ? 182 : (obj_id == 1) ? 92  : 197;
        uint8_t fill_b = (obj_id == 0) ? 212 : (obj_id == 1) ? 246 : 94;

        for (int y = y0; y < std::min(y0 + bh, FRAME_H); ++y) {
            for (int x = x0; x < std::min(x0 + bw, FRAME_W); ++x) {
                int idx = (y * FRAME_W + x) * 3;
                buf[idx + 0] = static_cast<uint8_t>(
                    std::clamp(buf[idx + 0] + fill_r / 4, 0, 255));
                buf[idx + 1] = static_cast<uint8_t>(
                    std::clamp(buf[idx + 1] + fill_g / 4, 0, 255));
                buf[idx + 2] = static_cast<uint8_t>(
                    std::clamp(buf[idx + 2] + fill_b / 4, 0, 255));
            }
        }

        // Draw bounding box edges (neon)
        for (int x = x0; x < std::min(x0 + bw, FRAME_W); ++x) {
            auto paint = [&](int py) {
                if (py < 0 || py >= FRAME_H) return;
                int idx = (py * FRAME_W + x) * 3;
                buf[idx + 0] = fill_r;
                buf[idx + 1] = fill_g;
                buf[idx + 2] = fill_b;
            };
            paint(y0); paint(y0 + 1);
            paint(std::min(y0 + bh - 1, FRAME_H - 1));
            paint(std::min(y0 + bh - 2, FRAME_H - 1));
        }
        for (int y = y0; y < std::min(y0 + bh, FRAME_H); ++y) {
            auto paint = [&](int px) {
                if (px < 0 || px >= FRAME_W) return;
                int idx = (y * FRAME_W + px) * 3;
                buf[idx + 0] = fill_r;
                buf[idx + 1] = fill_g;
                buf[idx + 2] = fill_b;
            };
            paint(x0); paint(x0 + 1);
            paint(std::min(x0 + bw - 1, FRAME_W - 1));
            paint(std::min(x0 + bw - 2, FRAME_W - 1));
        }
    }

    void draw_scanline(uint8_t* buf, int sy) {
        if (sy < 0 || sy >= FRAME_H) return;
        for (int x = 0; x < FRAME_W; ++x) {
            int idx = (sy * FRAME_W + x) * 3;
            buf[idx + 0] = std::min(255, buf[idx + 0] + 80);
            buf[idx + 1] = std::min(255, buf[idx + 1] + 160);
            buf[idx + 2] = std::min(255, buf[idx + 2] + 160);
        }
    }
};

// ─── TCP Notifier ─────────────────────────────────────────────────────────────
class TcpNotifier {
public:
    TcpNotifier() : sock_(-1) {}

    ~TcpNotifier() {
        if (sock_ >= 0) ::close(sock_);
    }

    bool connect_to_backend() {
        sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0) return false;

        // Allow reuse
        int opt = 1;
        ::setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(NOTIFY_PORT);
        ::inet_pton(AF_INET, NOTIFY_HOST, &addr.sin_addr);

        return ::connect(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    }

    // Send a compact 24-byte notification packet:
    // [ frame_id(8) | timestamp_ns(8) | fps(4) | flags(4) ]
    bool notify(uint64_t frame_id, int64_t ts_ns, float fps) {
        if (sock_ < 0) return false;
        uint8_t pkt[24];
        std::memcpy(pkt,      &frame_id, 8);
        std::memcpy(pkt + 8,  &ts_ns,    8);
        std::memcpy(pkt + 16, &fps,       4);
        uint32_t flags = 0;
        std::memcpy(pkt + 20, &flags,     4);
        ssize_t sent = ::send(sock_, pkt, sizeof(pkt), MSG_NOSIGNAL);
        return sent == sizeof(pkt);
    }

private:
    int sock_;
};

// ─── Shared Memory Manager ───────────────────────────────────────────────────
class ShmProducer {
public:
    ShmProducer() : shm_fd_(-1), shm_ptr_(nullptr), header_(nullptr), slots_(nullptr) {}

    ~ShmProducer() {
        cleanup();
    }

    void init() {
        // Remove any stale segment
        ::shm_unlink(SHM_NAME);

        shm_fd_ = ::shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
        if (shm_fd_ < 0)
            throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));

        if (::ftruncate(shm_fd_, static_cast<off_t>(SHM_SIZE)) < 0)
            throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));

        shm_ptr_ = ::mmap(nullptr, SHM_SIZE,
                          PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED)
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));

        // Advise kernel for sequential access patterns
        ::madvise(shm_ptr_, SHM_SIZE, MADV_SEQUENTIAL | MADV_WILLNEED);

        header_ = new (shm_ptr_) ShmHeader();
        header_->producer_pid.store(static_cast<uint32_t>(::getpid()));

        slots_  = reinterpret_cast<FrameSlot*>(
            static_cast<uint8_t*>(shm_ptr_) + sizeof(ShmHeader));

        std::cout << "[ShmProducer] Shared memory allocated: "
                  << SHM_SIZE / 1024 << " KB @ " << SHM_NAME << "\n"
                  << "              Ring slots  : " << RING_CAPACITY << "\n"
                  << "              Frame size  : " << FRAME_W << "x" << FRAME_H
                  << " (" << FRAME_BYTES / 1024 << " KB each)\n";
    }

    // Returns pointer to the next writable slot (spin-waits if ring is full)
    FrameSlot* acquire_write_slot() {
        while (true) {
            uint64_t wi = header_->write_idx.load(std::memory_order_relaxed);
            uint64_t ri = header_->read_idx.load(std::memory_order_acquire);
            if (wi - ri < static_cast<uint64_t>(RING_CAPACITY))
                return &slots_[wi % RING_CAPACITY];
            // Ring full — yield and retry
            std::this_thread::yield();
        }
    }

    void commit_write() {
        header_->write_idx.fetch_add(1, std::memory_order_release);
        header_->total_frames.fetch_add(1, std::memory_order_relaxed);
    }

    void update_fps(float fps) {
        header_->fps_actual.store(fps, std::memory_order_relaxed);
    }

    ShmHeader* header() { return header_; }

private:
    void cleanup() {
        if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
            ::munmap(shm_ptr_, SHM_SIZE);
            shm_ptr_ = nullptr;
        }
        if (shm_fd_ >= 0) {
            ::close(shm_fd_);
            shm_fd_ = -1;
        }
        ::shm_unlink(SHM_NAME);
    }

    int         shm_fd_;
    void*       shm_ptr_;
    ShmHeader*  header_;
    FrameSlot*  slots_;
};

// ─── Main Producer Loop ───────────────────────────────────────────────────────
int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n"
              << "║   EdgeAI-Sentinel-Nexus :: Frame Producer   ║\n"
              << "╚══════════════════════════════════════════════╝\n\n";

    ShmProducer   shm;
    FrameGenerator gen;
    TcpNotifier   notifier;

    try {
        shm.init();
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }

    // Try to connect to the Python backend (non-fatal if unavailable)
    bool connected = notifier.connect_to_backend();
    if (connected) {
        std::cout << "[TcpNotifier] Connected to backend at "
                  << NOTIFY_HOST << ":" << NOTIFY_PORT << "\n";
    } else {
        std::cout << "[TcpNotifier] Backend not reachable yet — "
                     "running in standalone mode.\n";
    }

    using Clock  = std::chrono::steady_clock;
    using NS     = std::chrono::nanoseconds;

    const NS frame_interval(static_cast<long>(1e9 / TARGET_FPS));
    auto next_frame_time = Clock::now();

    uint64_t frame_id   = 0;
    uint64_t fps_window = 0;
    auto     fps_start  = Clock::now();

    std::cout << "[Producer] Streaming at target " << TARGET_FPS << " FPS...\n\n";

    while (true) {
        // ── Precise frame pacing ──────────────────────────────────────────────
        auto now = Clock::now();
        if (now < next_frame_time) {
            // High-resolution busy-wait for the last ~100 µs; sleep the rest
            auto sleep_until = next_frame_time - std::chrono::microseconds(100);
            if (sleep_until > now)
                std::this_thread::sleep_until(sleep_until);
            while (Clock::now() < next_frame_time)
                std::this_thread::yield();
        }
        next_frame_time += frame_interval;

        // ── Acquire ring slot ─────────────────────────────────────────────────
        FrameSlot* slot = shm.acquire_write_slot();

        // ── Render frame ──────────────────────────────────────────────────────
        gen.render(slot->pixels, frame_id);

        // ── Fill metadata ─────────────────────────────────────────────────────
        int64_t ts_ns = std::chrono::duration_cast<NS>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        slot->meta.frame_id    = frame_id;
        slot->meta.timestamp_ns = ts_ns;
        slot->meta.width       = FRAME_W;
        slot->meta.height      = FRAME_H;
        slot->meta.channels    = FRAME_CHANNELS;
        slot->meta.byte_size   = FRAME_BYTES;

        // ── Commit to ring ────────────────────────────────────────────────────
        shm.commit_write();

        // ── Notify backend ────────────────────────────────────────────────────
        if (connected) {
            // FPS updated every 60 frames
            float fps = shm.header()->fps_actual.load(std::memory_order_relaxed);
            if (!notifier.notify(frame_id, ts_ns, fps)) {
                connected = false;   // connection lost
                std::cerr << "[TcpNotifier] Connection lost.\n";
            }
        }

        ++frame_id;
        ++fps_window;

        // ── FPS telemetry ─────────────────────────────────────────────────────
        if (fps_window >= 60) {
            auto elapsed = Clock::now() - fps_start;
            double elapsed_s = std::chrono::duration<double>(elapsed).count();
            float fps_actual = static_cast<float>(fps_window / elapsed_s);
            shm.update_fps(fps_actual);
            fps_start  = Clock::now();
            fps_window = 0;

            std::cout << "\r[Producer] frame=" << frame_id
                      << "  FPS=" << fps_actual
                      << "  total=" << shm.header()->total_frames.load()
                      << "          " << std::flush;
        }
    }

    return 0;
}

// fast_detector — native (C++/OpenVINO) implementation of the YOLO detection
// stage of scripts/vlm_pipeline.py.
//
// video -> ffmpeg (rawvideo pipe, every Nth frame) -> OpenVINO YOLO26 end2end ->
// event grouping -> candidate scoring -> best crops on disk + events.json
//
// The VLM stage stays in Python (scripts/fast_pipeline.py) — it is an HTTP call
// to llama.cpp and gains nothing from C++. To let Python run VLM in parallel
// with detection, --stream-events emits each event to stdout as it closes
// (JSONL: {"type":"event",...} per event, {"type":"summary",...} at the end);
// human-readable logging then goes to stderr. Without the flag, all events are
// written to --out-json at the end (used by the sequential mode and benchmark).
//
// Logic ported 1:1 from vlm_pipeline.py:
//   - find_matching_event (same-frame detections match by IoU only)
//   - candidate score = conf + area_score(0-0.3) + sharpness(0-0.4, Laplacian
//     var / 500) + clip_penalty(-0.15)
//   - keep_top_candidates with immediate file deletion of evicted candidates
//
// Known deliberate differences from the Python version:
//   - annotated frames contain plain rectangles, not ultralytics result.plot()
//   - inference runs on the OpenVINO IR export, so confidences may differ from
//     the torch best.pt path within float tolerance

#include <openvino/openvino.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb_image_write.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Cross-platform process pipe + null device. On Windows the rawvideo pipe must
// be opened in binary mode ("rb") or CRLF translation corrupts the byte stream.
#ifdef _WIN32
#  include <io.h>
#  define OV_POPEN  _popen
#  define OV_PCLOSE _pclose
#  define OV_DEVNULL "NUL"
#  define OV_PIPE_READ "rb"  // binary: avoid CRLF translation of the rawvideo stream
#else
#  define OV_POPEN  popen
#  define OV_PCLOSE pclose
#  define OV_DEVNULL "/dev/null"
#  define OV_PIPE_READ "r"   // POSIX pipes have no text/binary mode; glibc rejects "rb"
#endif

namespace {

// -----------------------------
// CLI args
// -----------------------------

struct Args {
    std::string video;
    std::string model = "models/best_openvino_model/best.xml";
    std::string device = "CPU";
    std::string tmp_dir = "_tmp_candidates";
    std::string out_json = "events.json";
    int frame_step = 10;
    float conf = 0.5f;
    int padding_x = 20;
    int padding_y = 12;
    float event_gap_sec = 3.0f;
    float event_iou_thr = 0.25f;
    float event_center_thr = 0.05f;
    float event_split_height_frac = 0.10f;  // plaque-swap guard (see find_matching_event)
    int keep_top_candidates = 3;
    long max_processed_frames = 0;
    bool no_save_images = false;
    bool quiet = false;
    bool stream_events = false;  // emit closed events as JSONL to stdout
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "fast_detector: %s\n", msg.c_str());
    std::exit(1);
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) die(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--video") a.video = need(i);
        else if (k == "--model") a.model = need(i);
        else if (k == "--device") a.device = need(i);
        else if (k == "--tmp-dir") a.tmp_dir = need(i);
        else if (k == "--out-json") a.out_json = need(i);
        else if (k == "--frame-step") a.frame_step = std::stoi(need(i));
        else if (k == "--conf") a.conf = std::stof(need(i));
        else if (k == "--padding-x") a.padding_x = std::stoi(need(i));
        else if (k == "--padding-y") a.padding_y = std::stoi(need(i));
        else if (k == "--event-gap-sec") a.event_gap_sec = std::stof(need(i));
        else if (k == "--event-iou-thr") a.event_iou_thr = std::stof(need(i));
        else if (k == "--event-center-thr") a.event_center_thr = std::stof(need(i));
        else if (k == "--event-split-height-frac") a.event_split_height_frac = std::stof(need(i));
        else if (k == "--keep-top-candidates") a.keep_top_candidates = std::stoi(need(i));
        else if (k == "--max-processed-frames") a.max_processed_frames = std::stol(need(i));
        else if (k == "--no-save-images") a.no_save_images = true;
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--stream-events") a.stream_events = true;
        else die("unknown argument: " + k);
    }
    if (a.video.empty()) die("--video is required");
    if (a.frame_step <= 0) die("--frame-step must be >= 1");
    if (a.keep_top_candidates <= 0) die("--keep-top-candidates must be >= 1");
    return a;
}

// -----------------------------
// Small helpers (ports of vlm_pipeline.py helpers)
// -----------------------------

struct Box { int x1, y1, x2, y2; };

int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

long box_area(const Box& b) {
    return (long)std::max(0, b.x2 - b.x1) * std::max(0, b.y2 - b.y1);
}

constexpr size_t RECENT_HEIGHTS_MAXLEN = 5;

int box_height(const Box& b) { return b.y2 - b.y1; }

// Median height of an event's recent detections — reference geometry for the
// plaque-swap guard. Mirrors vlm_pipeline.reference_height 1:1: sort ascending,
// take element at index n/2 (upper of the two middles for even counts), so both
// engines decide identically.
int reference_height(const std::vector<int>& heights) {
    if (heights.empty()) return 0;
    std::vector<int> s = heights;
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

void push_recent_height(std::vector<int>& heights, int h) {
    heights.push_back(h);
    if (heights.size() > RECENT_HEIGHTS_MAXLEN) heights.erase(heights.begin());
}

double box_iou(const Box& a, const Box& b) {
    int ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    int ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    long inter = (long)std::max(0, ix2 - ix1) * std::max(0, iy2 - iy1);
    long uni = box_area(a) + box_area(b) - inter;
    return uni > 0 ? (double)inter / (double)uni : 0.0;
}

double center_distance_norm(const Box& a, const Box& b, int w, int h) {
    double acx = (a.x1 + a.x2) / 2.0, acy = (a.y1 + a.y2) / 2.0;
    double bcx = (b.x1 + b.x2) / 2.0, bcy = (b.y1 + b.y2) / 2.0;
    double diag = std::max(1.0, std::sqrt((double)w * w + (double)h * h));
    return std::sqrt((acx - bcx) * (acx - bcx) + (acy - bcy) * (acy - bcy)) / diag;
}

Box expand_box(const Box& b, int w, int h, int px, int py) {
    return {clampi(b.x1 - px, 0, w - 1), clampi(b.y1 - py, 0, h - 1),
            clampi(b.x2 + px, 0, w - 1), clampi(b.y2 + py, 0, h - 1)};
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else out += (char)c;
        }
    }
    return out;
}

// Format a duration in seconds as HH:MM:SS.mmm, matching
// vlm_pipeline.seconds_to_timestamp so logs read consistently across engines.
std::string seconds_to_timestamp(double seconds) {
    if (seconds < 0) seconds = 0;
    int h = (int)(seconds / 3600);
    int m = (int)((seconds - h * 3600) / 60);
    double s = seconds - h * 3600 - m * 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%06.3f", h, m, s);
    return buf;
}

// -----------------------------
// Async image writer (single thread => deletes stay ordered after writes)
// -----------------------------

struct WriteTask {
    enum Kind { PNG, JPG, DEL } kind;
    std::string path;
    std::vector<uint8_t> bgr;  // interleaved BGR, empty for DEL
    int w = 0, h = 0;
};

class ImageWriter {
public:
    explicit ImageWriter(size_t max_queue = 16) : max_queue_(max_queue) {
        worker_ = std::thread([this] { run(); });
    }
    ~ImageWriter() { finish(); }

    void submit(WriteTask&& t) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_space_.wait(lk, [this] { return q_.size() < max_queue_; });
        q_.push_back(std::move(t));
        cv_data_.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (done_) return;
            done_ = true;
        }
        cv_data_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

private:
    static void write_bgr_as_rgb(WriteTask& t) {
        // stb expects RGB byte order
        for (size_t i = 0; i + 2 < t.bgr.size(); i += 3) std::swap(t.bgr[i], t.bgr[i + 2]);
        if (t.kind == WriteTask::PNG)
            stbi_write_png(t.path.c_str(), t.w, t.h, 3, t.bgr.data(), t.w * 3);
        else
            stbi_write_jpg(t.path.c_str(), t.w, t.h, 3, t.bgr.data(), 90);
    }

    void run() {
        for (;;) {
            WriteTask t;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_data_.wait(lk, [this] { return !q_.empty() || done_; });
                if (q_.empty()) return;
                t = std::move(q_.front());
                q_.pop_front();
                cv_space_.notify_one();
            }
            if (t.kind == WriteTask::DEL) std::remove(t.path.c_str());
            else write_bgr_as_rgb(t);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_data_, cv_space_;
    std::deque<WriteTask> q_;
    size_t max_queue_;
    bool done_ = false;
    std::thread worker_;
};

// Pool of writers. Tasks are routed by path hash, so a DEL for a file always
// lands on the same (single-threaded, FIFO) writer that wrote it.
class WriterPool {
public:
    explicit WriterPool(size_t n = 4) {
        for (size_t i = 0; i < n; ++i)
            writers_.emplace_back(new ImageWriter());
    }
    void submit(WriteTask&& t) {
        size_t idx = std::hash<std::string>{}(t.path) % writers_.size();
        writers_[idx]->submit(std::move(t));
    }
    void finish() {
        for (auto& w : writers_) w->finish();
    }
private:
    std::vector<std::unique_ptr<ImageWriter>> writers_;
};

// -----------------------------
// Video input via ffmpeg pipe
// -----------------------------

struct VideoInfo { int width = 0, height = 0; double fps = 30.0; long nb_frames = 0; };

#ifdef _WIN32
// cmd.exe quoting: wrap in double quotes and escape embedded double quotes.
std::string shell_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? std::string("\\\"") : std::string(1, c);
    out += "\"";
    return out;
}
#else
// POSIX /bin/sh quoting: wrap in single quotes.
std::string shell_quote(const std::string& s) {
    std::string out = "'";
    for (char c : s) out += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    out += "'";
    return out;
}
#endif

VideoInfo probe_video(const std::string& path) {
    std::string cmd =
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of default=nw=1 " + shell_quote(path) + " 2>" OV_DEVNULL;
    FILE* p = OV_POPEN(cmd.c_str(), "r");
    if (!p) die("failed to run ffprobe");
    VideoInfo vi;
    char line[256];
    while (std::fgets(line, sizeof(line), p)) {
        std::string s(line);
        auto val = [&](const char* key) -> std::optional<std::string> {
            std::string k = std::string(key) + "=";
            if (s.rfind(k, 0) == 0) {
                std::string v = s.substr(k.size());
                while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
                return v;
            }
            return std::nullopt;
        };
        if (auto v = val("width")) vi.width = std::stoi(*v);
        else if (auto v = val("height")) vi.height = std::stoi(*v);
        else if (auto v = val("nb_frames")) { if (*v != "N/A") vi.nb_frames = std::stol(*v); }
        else if (auto v = val("r_frame_rate")) {
            double num = 0, den = 1;
            if (std::sscanf(v->c_str(), "%lf/%lf", &num, &den) == 2 && den > 0) vi.fps = num / den;
            else vi.fps = std::atof(v->c_str());
        }
    }
    OV_PCLOSE(p);
    if (vi.width <= 0 || vi.height <= 0) die("ffprobe could not read video: " + path);
    if (vi.fps <= 0) vi.fps = 30.0;
    return vi;
}

// Reader thread: decodes via ffmpeg child process, hands out sampled frames.
class FrameReader {
public:
    FrameReader(const std::string& video, int step, int w, int h, size_t depth = 3)
        : frame_bytes_((size_t)w * h * 3), depth_(depth) {
        std::ostringstream cmd;
        cmd << "ffmpeg -v error -i " << shell_quote(video)
            << " -vf \"select=not(mod(n\\," << step << "))\" -fps_mode vfr"
            << " -f rawvideo -pix_fmt bgr24 - 2>" OV_DEVNULL;
        // Binary read mode on Windows (CRLF translation would corrupt the
        // rawvideo byte stream); plain "r" on POSIX, which glibc requires.
        pipe_ = OV_POPEN(cmd.str().c_str(), OV_PIPE_READ);
        if (!pipe_) die("failed to start ffmpeg");
        worker_ = std::thread([this] { run(); });
    }

    ~FrameReader() {
        stop_.store(true);
        cv_space_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (pipe_) OV_PCLOSE(pipe_);
    }

    // Returns false at end of stream.
    bool next(std::vector<uint8_t>& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_data_.wait(lk, [this] { return !q_.empty() || eof_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        cv_space_.notify_one();
        return true;
    }

private:
    void run() {
        for (;;) {
            std::vector<uint8_t> buf(frame_bytes_);
            size_t got = std::fread(buf.data(), 1, frame_bytes_, pipe_);
            if (got != frame_bytes_) break;
            std::unique_lock<std::mutex> lk(mu_);
            cv_space_.wait(lk, [this] { return q_.size() < depth_ || stop_.load(); });
            if (stop_.load()) break;
            q_.push_back(std::move(buf));
            cv_data_.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu_);
        eof_ = true;
        cv_data_.notify_all();
    }

    FILE* pipe_ = nullptr;
    size_t frame_bytes_, depth_;
    std::mutex mu_;
    std::condition_variable cv_data_, cv_space_;
    std::deque<std::vector<uint8_t>> q_;
    bool eof_ = false;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

// -----------------------------
// Preprocess: BGR frame -> letterboxed RGB CHW float (ultralytics LetterBox)
// -----------------------------

struct Letterbox { double r; int left, top; };

Letterbox letterbox_into(const uint8_t* bgr, int w, int h, int sw, int sh, float* chw) {
    double r = std::min((double)sw / w, (double)sh / h);
    int nw = (int)std::lround(w * r), nh = (int)std::lround(h * r);
    double dw = (sw - nw) / 2.0, dh = (sh - nh) / 2.0;
    int left = (int)std::lround(dw - 0.1), top = (int)std::lround(dh - 0.1);

    const float pad = 114.0f / 255.0f;
    std::fill(chw, chw + (size_t)3 * sw * sh, pad);

    float* rp = chw;                       // RGB planes
    float* gp = chw + (size_t)sw * sh;
    float* bp = chw + (size_t)2 * sw * sh;

    // bilinear resize of the w*h source into the nw*nh region at (left, top)
    double sx = (double)w / nw, sy = (double)h / nh;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < nh; ++y) {
        double fy = (y + 0.5) * sy - 0.5;
        int y0 = (int)std::floor(fy);
        double wy = fy - y0;
        int y1 = std::min(y0 + 1, h - 1);
        y0 = std::max(y0, 0);
        size_t row = (size_t)(top + y) * sw + left;
        const uint8_t* r0 = bgr + (size_t)y0 * w * 3;
        const uint8_t* r1 = bgr + (size_t)y1 * w * 3;
        for (int x = 0; x < nw; ++x) {
            double fx = (x + 0.5) * sx - 0.5;
            int x0 = (int)std::floor(fx);
            double wx = fx - x0;
            int x1 = std::min(x0 + 1, w - 1);
            x0 = std::max(x0, 0);
            double w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx;
            double w10 = wy * (1 - wx), w11 = wy * wx;
            const uint8_t* p00 = r0 + (size_t)x0 * 3;
            const uint8_t* p01 = r0 + (size_t)x1 * 3;
            const uint8_t* p10 = r1 + (size_t)x0 * 3;
            const uint8_t* p11 = r1 + (size_t)x1 * 3;
            double b = w00 * p00[0] + w01 * p01[0] + w10 * p10[0] + w11 * p11[0];
            double g = w00 * p00[1] + w01 * p01[1] + w10 * p10[1] + w11 * p11[1];
            double rr = w00 * p00[2] + w01 * p01[2] + w10 * p10[2] + w11 * p11[2];
            rp[row + x] = (float)(rr / 255.0);
            gp[row + x] = (float)(g / 255.0);
            bp[row + x] = (float)(b / 255.0);
        }
    }
    return {r, left, top};
}

// -----------------------------
// Sharpness: cv2.Laplacian(gray, CV_64F).var() with ksize=1, BORDER_REFLECT_101
// -----------------------------

double laplacian_variance(const uint8_t* bgr, int stride_px, int w, int h) {
    if (w <= 0 || h <= 0) return 0.0;
    std::vector<int16_t> gray((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = bgr + (size_t)y * stride_px * 3;
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = row + (size_t)x * 3;
            // OpenCV BGR2GRAY fixed-point: (B*1868 + G*9617 + R*4899 + 8192) >> 14
            gray[(size_t)y * w + x] =
                (int16_t)((p[0] * 1868 + p[1] * 9617 + p[2] * 4899 + 8192) >> 14);
        }
    }
    auto ref = [](int i, int n) {  // BORDER_REFLECT_101
        if (n == 1) return 0;
        if (i < 0) return -i;
        if (i >= n) return 2 * n - i - 2;
        return i;
    };
    double sum = 0, sum2 = 0;
    for (int y = 0; y < h; ++y) {
        int yu = ref(y - 1, h), yd = ref(y + 1, h);
        for (int x = 0; x < w; ++x) {
            int xl = ref(x - 1, w), xr = ref(x + 1, w);
            double v = (double)gray[(size_t)yu * w + x] + gray[(size_t)yd * w + x] +
                       gray[(size_t)y * w + xl] + gray[(size_t)y * w + xr] -
                       4.0 * gray[(size_t)y * w + x];
            sum += v;
            sum2 += v * v;
        }
    }
    double n = (double)w * h;
    double mean = sum / n;
    return sum2 / n - mean * mean;
}

// -----------------------------
// Event structures (port of DonationEvent / CandidateCrop)
// -----------------------------

struct Candidate {
    int frame_idx = 0;
    double timestamp_sec = 0;
    float confidence = 0;
    int class_id = 0;
    Box base_box{}, padded_box{};
    std::string crop_path, ann_path, orig_path;  // ann/orig empty when --no-save-images
    double score = 0;
};

struct Event {
    int event_id = 0;
    double start_sec = 0, end_sec = 0;
    int first_frame = 0, last_frame = 0;
    Box last_box{};
    int detections_count = 0;
    float best_confidence = 0;
    double best_timestamp_sec = 0;
    int best_frame_idx = 0;
    bool emitted = false;  // streaming: already sent to stdout, don't resend
    std::vector<Candidate> candidates;
    std::vector<int> recent_heights;  // reference geometry for the plaque-swap guard
};

// Port of vlm_pipeline.find_matching_event, including the plaque-swap guard
// (split_height_frac > 0): two donations shown back-to-back in the same overlay
// slot overlap heavily (high IoU), so geometry alone merges them and the second is
// lost; but the box height jumps when the plaque is replaced (different message
// length) while staying ~stable within one donation. A candidate whose height
// differs from the event's reference height by more than split_height_frac is
// treated as a DIFFERENT plaque and does not match. The decision uses only integer
// box coords (identical in both engines), so cpp/py output stays in sync.
Event* find_matching_event(std::vector<Event*>& events, const Box& box, int w, int h,
                           float iou_thr, float center_thr, int current_frame_idx,
                           float split_height_frac) {
    Event* best = nullptr;
    double best_score = -999.0;
    int box_h = box_height(box);
    for (Event* ev : events) {
        double iou = box_iou(ev->last_box, box);
        double dist = center_distance_norm(ev->last_box, box, w, h);
        if (ev->last_frame == current_frame_idx) {
            if (iou < iou_thr) continue;  // same frame: IoU only
        } else {
            if (iou < iou_thr && dist > center_thr) continue;
        }
        if (split_height_frac > 0.0f) {
            int ref_h = reference_height(ev->recent_heights);
            if (ref_h > 0 && std::abs(box_h - ref_h) > split_height_frac * ref_h)
                continue;  // plaque height changed → different donation, don't merge
        }
        double score = iou - dist;
        if (score > best_score) { best_score = score; best = ev; }
    }
    return best;
}

void delete_candidate_files(WriterPool& writer, const Candidate& c) {
    for (const std::string* p : {&c.crop_path, &c.ann_path, &c.orig_path})
        if (!p->empty()) writer.submit({WriteTask::DEL, *p, {}, 0, 0});
}

void add_candidate(WriterPool& writer, Event& ev, Candidate&& c, int max_candidates) {
    ev.candidates.push_back(std::move(c));
    std::stable_sort(ev.candidates.begin(), ev.candidates.end(),
                     [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    if ((int)ev.candidates.size() > max_candidates) {
        delete_candidate_files(writer, ev.candidates[max_candidates]);
        ev.candidates.resize(max_candidates);
    }
}

void trim_to_best(WriterPool& writer, Event& ev) {
    for (size_t i = 1; i < ev.candidates.size(); ++i)
        delete_candidate_files(writer, ev.candidates[i]);
    if (ev.candidates.size() > 1) ev.candidates.resize(1);
}

void draw_box(std::vector<uint8_t>& bgr, int w, int h, const Box& b, int thick = 3) {
    auto set_px = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        uint8_t* p = bgr.data() + ((size_t)y * w + x) * 3;
        p[0] = 56; p[1] = 56; p[2] = 255;  // ultralytics-style red
    };
    for (int t = 0; t < thick; ++t) {
        for (int x = b.x1 - t; x <= b.x2 + t; ++x) { set_px(x, b.y1 - t); set_px(x, b.y2 + t); }
        for (int y = b.y1 - t; y <= b.y2 + t; ++y) { set_px(b.x1 - t, y); set_px(b.x2 + t, y); }
    }
}

// Serialize one event as a compact JSON object. Shared by the final events.json
// array and the streaming protocol, so both stay in sync.
void write_event_obj(std::ostream& os, const Event& ev) {
    os << "{\"event_id\": " << ev.event_id
       << ", \"start_sec\": " << ev.start_sec
       << ", \"end_sec\": " << ev.end_sec
       << ", \"first_frame\": " << ev.first_frame
       << ", \"last_frame\": " << ev.last_frame
       << ", \"detections_count\": " << ev.detections_count
       << ", \"best_confidence\": " << ev.best_confidence
       << ", \"best_timestamp_sec\": " << ev.best_timestamp_sec
       << ", \"best_frame_idx\": " << ev.best_frame_idx
       << ", \"candidates\": [";
    for (size_t j = 0; j < ev.candidates.size(); ++j) {
        const Candidate& c = ev.candidates[j];
        if (j) os << ", ";
        os << "{\"frame_idx\": " << c.frame_idx
           << ", \"timestamp_sec\": " << c.timestamp_sec
           << ", \"confidence\": " << c.confidence
           << ", \"class_id\": " << c.class_id
           << ", \"score\": " << c.score
           << ", \"base_box\": [" << c.base_box.x1 << ", " << c.base_box.y1 << ", "
           << c.base_box.x2 << ", " << c.base_box.y2 << "]"
           << ", \"padded_box\": [" << c.padded_box.x1 << ", " << c.padded_box.y1 << ", "
           << c.padded_box.x2 << ", " << c.padded_box.y2 << "]"
           << ", \"crop_path\": \"" << json_escape(c.crop_path) << "\""
           << ", \"annotated_frame_path\": "
           << (c.ann_path.empty() ? "null" : "\"" + json_escape(c.ann_path) + "\"")
           << ", \"original_frame_path\": "
           << (c.orig_path.empty() ? "null" : "\"" + json_escape(c.orig_path) + "\"")
           << "}";
    }
    os << "]}";
}

}  // namespace

// -----------------------------
// Main
// -----------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    auto t_total0 = std::chrono::steady_clock::now();

    // tmp files are short-lived; trade compression ratio for encode speed
    stbi_write_png_compression_level = 2;

    std::string model_xml = args.model;
    if (model_xml.size() < 4 || model_xml.substr(model_xml.size() - 4) != ".xml")
        model_xml += "/best.xml";

    VideoInfo vi = probe_video(args.video);

    ov::Core core;
    auto compile_on = [&](const std::string& dev) {
        return core.compile_model(
            model_xml, dev,
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    };
    ov::CompiledModel compiled;
    try {
        compiled = compile_on(args.device);
    } catch (const std::exception& e) {
        if (args.device != "CPU") {
            std::fprintf(stderr,
                         "fast_detector: device '%s' unavailable (%s); falling back to CPU\n",
                         args.device.c_str(), e.what());
            args.device = "CPU";
            compiled = compile_on("CPU");
        } else {
            throw;
        }
    }
    uint32_t nireq = 4;
    try {
        nireq = compiled.get_property(ov::optimal_number_of_infer_requests);
    } catch (...) {}
    nireq = std::max(1u, std::min(nireq, 8u));
    ov::Shape in_shape = compiled.input().get_shape();  // [1,3,SH,SW]
    int SH = (int)in_shape[2];   // network input height
    int SW = (int)in_shape[3];   // network input width (may differ -> rect input)

    std::string video_name = args.video;
    if (auto pos = video_name.find_last_of('/'); pos != std::string::npos)
        video_name = video_name.substr(pos + 1);

    // When streaming, stdout carries the machine-readable JSONL protocol, so all
    // human-readable logging goes to stderr instead.
    FILE* logf = args.stream_events ? stderr : stdout;

    if (!args.quiet) {
        std::fprintf(logf, "fast_detector: %s %dx%d @ %.3f fps, %ld frames, step %d, device %s, input %dx%d\n",
                     video_name.c_str(), vi.width, vi.height, vi.fps, vi.nb_frames,
                     args.frame_step, args.device.c_str(), SW, SH);
        std::fflush(logf);
    }

    WriterPool writer;
    FrameReader reader(args.video, args.frame_step, vi.width, vi.height);

    // Async pipeline: a ring of nireq slots. Frames are submitted with
    // start_async and drained strictly in submission order, so the
    // (order-dependent) event grouping behaves exactly like the sync version.
    struct Slot {
        ov::InferRequest req;
        std::vector<float> input;
        std::vector<uint8_t> frame;
        long frame_idx = 0;
        Letterbox lb{};
    };
    std::vector<Slot> slots(nireq);
    for (auto& s : slots) {
        s.req = compiled.create_infer_request();
        s.input.resize((size_t)3 * SW * SH);
        s.req.set_input_tensor(ov::Tensor(ov::element::f32, in_shape, s.input.data()));
    }

    std::deque<Event> all_events;  // deque: stable element addresses for `active` pointers
    std::vector<Event*> active;

    long processed = 0, submitted = 0, raw_detections = 0, det_counter = 0;
    double t_infer = 0, t_read = 0;
    const int w = vi.width, h = vi.height;

    auto t_stage0 = std::chrono::steady_clock::now();
    auto batch_t0 = t_stage0;  // wall time since last progress line

    size_t head = 0, tail = 0, in_flight = 0;
    bool eof = false;

    while (true) {
        // keep the pipeline full
        while (!eof && in_flight < nireq) {
            if (args.max_processed_frames && submitted >= args.max_processed_frames) {
                eof = true;
                break;
            }
            Slot& s = slots[tail];
            auto tr0 = std::chrono::steady_clock::now();
            if (!reader.next(s.frame)) {
                eof = true;
                break;
            }
            t_read += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
            s.frame_idx = submitted * args.frame_step;
            ++submitted;
            s.lb = letterbox_into(s.frame.data(), w, h, SW, SH, s.input.data());
            s.req.start_async();
            tail = (tail + 1) % nireq;
            ++in_flight;
        }
        if (in_flight == 0) break;

        // drain the oldest slot
        Slot& slot = slots[head];
        head = (head + 1) % nireq;
        --in_flight;

        auto ti0 = std::chrono::steady_clock::now();
        slot.req.wait();
        t_infer += std::chrono::duration<double>(std::chrono::steady_clock::now() - ti0).count();

        const std::vector<uint8_t>& frame = slot.frame;
        const Letterbox lb = slot.lb;
        long frame_idx = slot.frame_idx;
        ++processed;
        double timestamp = frame_idx / vi.fps;

        // close events that exceeded the gap
        std::vector<Event*> still_active;
        for (Event* ev : active) {
            if (timestamp - ev->end_sec <= args.event_gap_sec) {
                still_active.push_back(ev);
            } else {
                trim_to_best(writer, *ev);
                if (args.stream_events) {
                    std::cout << "{\"type\": \"event\", \"event\": ";
                    write_event_obj(std::cout, *ev);
                    std::cout << "}\n" << std::flush;
                    ev->emitted = true;
                }
            }
        }
        active = std::move(still_active);

        // output: [1, 300, 6] = x1,y1,x2,y2,conf,cls in letterbox coords
        ov::Tensor out = slot.req.get_output_tensor();
        const float* det = out.data<const float>();
        size_t n_det = out.get_shape()[1];

        std::vector<std::tuple<Box, float, int>> boxes;
        for (size_t i = 0; i < n_det; ++i) {
            const float* d = det + i * 6;
            float conf = d[4];
            if (conf < args.conf) continue;
            int x1 = clampi((int)((d[0] - lb.left) / lb.r), 0, w - 1);
            int y1 = clampi((int)((d[1] - lb.top) / lb.r), 0, h - 1);
            int x2 = clampi((int)((d[2] - lb.left) / lb.r), 0, w - 1);
            int y2 = clampi((int)((d[3] - lb.top) / lb.r), 0, h - 1);
            boxes.push_back({Box{x1, y1, x2, y2}, conf, (int)d[5]});
        }

        if (!boxes.empty()) {
            std::vector<uint8_t> annotated;
            if (!args.no_save_images) {
                annotated = frame;
                for (auto& [bb, cf, cl] : boxes) draw_box(annotated, w, h, bb);
            }

            for (auto& [base_box, conf, cls_id] : boxes) {
                Box padded = expand_box(base_box, w, h, args.padding_x, args.padding_y);
                int cw = padded.x2 - padded.x1, ch = padded.y2 - padded.y1;
                if (cw <= 0 || ch <= 0) continue;

                std::vector<uint8_t> crop((size_t)cw * ch * 3);
                for (int y = 0; y < ch; ++y)
                    std::memcpy(crop.data() + (size_t)y * cw * 3,
                                frame.data() + ((size_t)(padded.y1 + y) * w + padded.x1) * 3,
                                (size_t)cw * 3);

                double area_score = std::min((double)box_area(base_box) /
                                             std::max(1L, (long)w * h) * 10.0, 0.3);
                double sharp = laplacian_variance(crop.data(), cw, cw, ch);
                double sharpness_score = std::min(sharp / 500.0, 0.4);
                bool clipped =
                    (padded.x1 == 0 && base_box.x1 > args.padding_x) ||
                    (padded.y1 == 0 && base_box.y1 > args.padding_y) ||
                    (padded.x2 == w - 1 && base_box.x2 < w - 1 - args.padding_x) ||
                    (padded.y2 == h - 1 && base_box.y2 < h - 1 - args.padding_y);
                double score = conf + area_score + sharpness_score + (clipped ? -0.15 : 0.0);

                char name[64];
                std::snprintf(name, sizeof(name), "/det%07ld_crop.png", det_counter);
                Candidate c;
                c.crop_path = args.tmp_dir + name;
                writer.submit({WriteTask::PNG, c.crop_path, crop, cw, ch});
                if (!args.no_save_images) {
                    std::snprintf(name, sizeof(name), "/det%07ld_ann.jpg", det_counter);
                    c.ann_path = args.tmp_dir + name;
                    writer.submit({WriteTask::JPG, c.ann_path, annotated, w, h});
                    std::snprintf(name, sizeof(name), "/det%07ld_orig.png", det_counter);
                    c.orig_path = args.tmp_dir + name;
                    writer.submit({WriteTask::PNG, c.orig_path, frame, w, h});
                }
                ++det_counter;

                Event* matched = find_matching_event(active, base_box, w, h,
                                                     args.event_iou_thr, args.event_center_thr,
                                                     (int)frame_idx, args.event_split_height_frac);
                if (!matched) {
                    all_events.push_back(Event{});
                    matched = &all_events.back();
                    matched->event_id = (int)all_events.size();
                    matched->start_sec = timestamp;
                    matched->first_frame = (int)frame_idx;
                    active.push_back(matched);
                }
                matched->end_sec = timestamp;
                matched->last_frame = (int)frame_idx;
                matched->last_box = base_box;
                matched->detections_count += 1;
                push_recent_height(matched->recent_heights, box_height(base_box));
                if (conf >= matched->best_confidence) {
                    matched->best_confidence = conf;
                    matched->best_timestamp_sec = timestamp;
                    matched->best_frame_idx = (int)frame_idx;
                }

                c.frame_idx = (int)frame_idx;
                c.timestamp_sec = timestamp;
                c.confidence = conf;
                c.class_id = cls_id;
                c.base_box = base_box;
                c.padded_box = padded;
                c.score = score;
                add_candidate(writer, *matched, std::move(c), args.keep_top_candidates);
                ++raw_detections;
            }
        }

        if (!args.quiet && processed % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double batch_elapsed = std::chrono::duration<double>(now - batch_t0).count();
            batch_t0 = now;
            std::fprintf(logf,
                         "Processed sampled frames: %ld, source frame: %ld, detections: %ld, events: %zu, "
                         "batch time: %.2fс\n",
                         processed, frame_idx, raw_detections, all_events.size(), batch_elapsed);
            std::fflush(logf);
        }
    }

    // Flush events still open at end of video.
    for (Event* ev : active) {
        trim_to_best(writer, *ev);
        if (args.stream_events) {
            std::cout << "{\"type\": \"event\", \"event\": ";
            write_event_obj(std::cout, *ev);
            std::cout << "}\n" << std::flush;
            ev->emitted = true;
        }
    }
    writer.finish();

    double yolo_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_stage0).count();
    double total_elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total0).count();

    // Header fields shared by events.json and the streaming summary line.
    auto write_header_fields = [&](std::ostream& os, const char* sep) {
        os << "\"video_name\": \"" << json_escape(video_name) << "\"," << sep
           << "\"fps\": " << vi.fps << "," << sep
           << "\"total_frames\": " << vi.nb_frames << "," << sep
           << "\"frame_width\": " << w << "," << sep
           << "\"frame_height\": " << h << "," << sep
           << "\"sampled_frames_processed\": " << processed << "," << sep
           << "\"raw_detections\": " << raw_detections << "," << sep
           << "\"events_count\": " << all_events.size() << "," << sep
           << "\"yolo_elapsed_sec\": " << yolo_elapsed << "," << sep
           << "\"total_elapsed_sec\": " << total_elapsed << "," << sep
           << "\"infer_sec\": " << t_infer << "," << sep
           << "\"decode_wait_sec\": " << t_read << "," << sep
           << "\"device\": \"" << json_escape(args.device) << "\"";
    };

    if (args.stream_events) {
        // All events were already streamed as they closed; close with a summary.
        std::cout << "{\"type\": \"summary\", ";
        write_header_fields(std::cout, " ");
        std::cout << "}\n" << std::flush;
    } else {
        std::ofstream js(args.out_json);
        if (!js) die("cannot write " + args.out_json);
        js << "{\n  ";
        write_header_fields(js, "\n  ");
        js << ",\n  \"events\": [\n";
        for (size_t i = 0; i < all_events.size(); ++i) {
            js << "    ";
            write_event_obj(js, all_events[i]);
            js << (i + 1 < all_events.size() ? "," : "") << "\n";
        }
        js << "  ]\n}\n";
        js.close();
    }

    if (!args.quiet) {
        std::fprintf(logf, "\nYOLO stage done in %s (infer %s, decode wait %s).\n",
                     seconds_to_timestamp(yolo_elapsed).c_str(),
                     seconds_to_timestamp(t_infer).c_str(),
                     seconds_to_timestamp(t_read).c_str());
        std::fprintf(logf, "Sampled frames: %ld, detections: %ld, events: %zu\n",
                     processed, raw_detections, all_events.size());
    }
    return 0;
}

#pragma once
// audio-frames.h: the engine's audio clock (cadenza-audio #1093) -- the one
// sample rate every stage runs at, and the VAE's hop: the samples per latent
// frame, 25 of them a second. The conversions between samples, latent frames
// and seconds are written here once, not at each log line and buffer size.

#include <limits>

// Samples per second, per channel: what the VAE decodes to and encodes from.
inline constexpr int ACE_SAMPLE_RATE = 48000;

// Samples per latent frame: the VAE's down- and upsampling (2x4x4x6x10).
inline constexpr int ACE_LATENT_HOP = 1920;

// The most latent frames a run holds: the length of the DiT's silence latent
// (silence_full), which pads and backs every context -- 600 s (#1139).
inline constexpr int ACE_MAX_LATENT_FRAMES = 15000;

// Latent frames per second (25, exactly).
inline constexpr float ACE_LATENT_FPS = (float) ACE_SAMPLE_RATE / ACE_LATENT_HOP;

static_assert(ACE_SAMPLE_RATE % ACE_LATENT_HOP == 0, "a whole number of latent frames a second");

// A latent frame count's length in seconds.
inline constexpr float latent_frames_sec(int frames) {
    return (float) frames * ACE_LATENT_HOP / ACE_SAMPLE_RATE;
}

// A sample count's (per channel) length in seconds.
inline constexpr float samples_sec(int samples) {
    return (float) samples / ACE_SAMPLE_RATE;
}

// The frames of slack a seconds-to-frame step gives a time meant to land on a
// frame start. A float k/25 s can sit just below it -- 2.12f is 2.1199999, and
// at 1440 s a float is off by up to ~0.0015 frames -- and a plain truncation
// then drops to frame k-1: 2991 of the 36000 starts to 1440 s did (#1129).
// 0.01 frames is 0.4 ms.
inline constexpr double ACE_FRAME_SLACK = 0.01;

// A frame count in double as an int, held to what one carries: a NaN is 0, a
// count past +-1e9 frames that bound -- so a request's repainting_end of 1e10 s
// is a far frame, not an undefined cast (#1129).
inline constexpr int frame_index(double frames) {
    if (frames != frames) {
        return 0;
    }
    if (frames < -1e9) {
        return -1000000000;
    }
    if (frames > 1e9) {
        return 1000000000;
    }
    return (int) frames;
}

// A count in double rounded down to an int, held as frame_index holds it.
inline constexpr int floor_index(double v) {
    const int whole = frame_index(v);
    return v < whole ? whole - 1 : whole;
}

// A count in double rounded up to an int, held as frame_index holds it.
inline constexpr int ceil_index(double v) {
    const int whole = frame_index(v);
    return v > whole ? whole + 1 : whole;
}

// The latent frame a time in seconds falls in: rounded down, a frame start
// counted as its own frame. A run's length, and a region's first frame (#1129).
inline constexpr int sec_latent_frame(float sec) {
    return floor_index((double) sec * ACE_LATENT_FPS + ACE_FRAME_SLACK);
}

// The frame a span ending at a time in seconds ends before: rounded up, so a
// region holds every frame it reaches into -- a frame end counted as the end,
// not one frame more. With sec_latent_frame for its start, a region's frames
// cover it whole: a sub-frame region, or one ending inside a pad rounded up to
// a frame, is not left short of its last frame (#1129).
inline constexpr int sec_latent_frame_end(float sec) {
    return ceil_index((double) sec * ACE_LATENT_FPS - ACE_FRAME_SLACK);
}

// The sample nearest a time in seconds, per channel: a pad's length in samples,
// and where a waveform splice cuts. Nearest, not truncated: a float time is
// off by up to ~3 samples at 1440 s, either way (#1129).
inline constexpr int sec_samples_nearest(float sec) {
    return floor_index((double) sec * ACE_SAMPLE_RATE + 0.5);
}

// Samples rounded down, and up, to a whole number of latent frames.
inline constexpr int samples_frames_floor(int samples) {
    return samples / ACE_LATENT_HOP;
}

inline constexpr int samples_frames_ceil(int samples) {
    return (int) (((long long) samples + ACE_LATENT_HOP - 1) / ACE_LATENT_HOP);
}

// A padded source's whole frames, rounded up -- what an audio outpaint's pad
// runs on to at most -- from its parts in samples, summed in 64 bits (#1139).
inline constexpr long long padded_samples_frames(long long lpad, long long len, long long rpad) {
    return (lpad + len + rpad + ACE_LATENT_HOP - 1) / ACE_LATENT_HOP;
}

// The whole number of latent frames nearest a span in seconds: a left pad's
// length, which places the source as near as a frame can to where it was asked.
inline constexpr int sec_latent_frames_nearest(float sec) {
    return floor_index((double) sec * ACE_LATENT_FPS + 0.5);
}

// The conversions, held at compile time: a second is 25 frames and 48000
// samples, and a frame start is its own frame -- 2.12 s frame 53, not 52.
static_assert(ACE_LATENT_FPS == 25.0f, "25 latent frames a second");
static_assert(latent_frames_sec(25) == 1.0f && latent_frames_sec(50) == 2.0f, "frames to seconds");
static_assert(samples_sec(ACE_SAMPLE_RATE) == 1.0f, "samples to seconds");
static_assert(sec_latent_frame(1.0f) == 25 && sec_latent_frame(0.039f) == 0 && sec_latent_frame(0.04f) == 1 &&
                  sec_latent_frame(2.12f) == 53 && sec_latent_frame(2.6f) == 65 && sec_latent_frame(1439.96f) == 35999,
              "seconds to the frame they fall in");
static_assert(sec_latent_frame_end(20.0f) == 500 && sec_latent_frame_end(10.03f) == 251 &&
                  sec_latent_frame_end(0.03f) == 1 && sec_latent_frame_end(0.0f) == 0 &&
                  sec_latent_frame_end(-0.5f) == -12,
              "a span's end, rounded up to the frame it reaches into");
static_assert(sec_latent_frames_nearest(0.03f) == 1 && sec_latent_frames_nearest(0.01f) == 0 &&
                  sec_latent_frames_nearest(-0.05f) == -1,
              "a pad's frames, nearest either side of 0");
static_assert(sec_latent_frame(-0.5f) == -13 && sec_latent_frame(-0.04f) == -1, "a time before 0 rounds down too");
static_assert(sec_samples_nearest(0.01f) == 480 && sec_samples_nearest(2.12f) == 101760 &&
                  sec_samples_nearest(-0.01f) == -480,
              "seconds to the nearest sample");

// Whether a run of this many latent frames is past a run's most -- the one
// test every bound on a run's frames makes (cadenza-audio #1139, #1144).
inline constexpr bool frames_past_run_most(long long frames) {
    return frames > ACE_MAX_LATENT_FRAMES;
}

static_assert(samples_frames_floor(97440) == 50 && samples_frames_ceil(97440) == 51 && samples_frames_ceil(96000) == 50,
              "samples to whole frames");
static_assert(!frames_past_run_most(ACE_MAX_LATENT_FRAMES) && frames_past_run_most(ACE_MAX_LATENT_FRAMES + 1),
              "a run of the most frames runs, one past it does not");
static_assert(samples_frames_floor(ACE_MAX_LATENT_FRAMES * ACE_LATENT_HOP + ACE_LATENT_HOP - 1) ==
                  ACE_MAX_LATENT_FRAMES,
              "600 s and a part frame of audio is the most whole frames, not past it");
static_assert(padded_samples_frames(0, 28800000, 0) == ACE_MAX_LATENT_FRAMES &&
                  padded_samples_frames(0, 28800001, 0) == ACE_MAX_LATENT_FRAMES + 1 &&
                  padded_samples_frames(1440, 96000, 480) == 51,
              "a 600 s source is a run's most frames, and not one past them");
static_assert(frame_index(1e12) == 1000000000 && frame_index(-1e12) == -1000000000 &&
                  frame_index(std::numeric_limits<double>::quiet_NaN()) == 0,
              "a frame count held to an int");

// The most samples per channel a planar buffer holds: half an int's range, so
// a stereo buffer's int indexing -- interleaved [t * 2], planar [T + t] --
// never overflows (cadenza-audio #1266). 6.2 hours at 48 kHz.
inline constexpr int ACE_MAX_AUDIO_SAMPLES = std::numeric_limits<int>::max() / 2;

// The samples of a source a run takes whole: its whole frames at most a run's
// most, and a part frame -- the floor the engine counts a source's frames by
// (SynthInput::frames, cadenza-audio #1144).
inline constexpr long long ACE_MAX_RUN_SOURCE_SAMPLES = (long long) (ACE_MAX_LATENT_FRAMES + 1) * ACE_LATENT_HOP - 1;

static_assert(samples_frames_floor((int) ACE_MAX_RUN_SOURCE_SAMPLES) == ACE_MAX_LATENT_FRAMES &&
                  samples_frames_floor((int) ACE_MAX_RUN_SOURCE_SAMPLES + 1) == ACE_MAX_LATENT_FRAMES + 1,
              "a run's source at its most, and one sample past it");

// The slack a VAE output buffer is sized with past its exact length: an
// encode's latent buffer 64 frames (its tiles' rounded trims), a decode's
// audio buffer 4096 samples. Every tool sizes by these (cadenza-audio #1266).
inline constexpr int ACE_VAE_ENC_SLACK_FRAMES  = 64;
inline constexpr int ACE_VAE_DEC_SLACK_SAMPLES = 4096;

// The most frames a latent decodes in one buffer: its samples and the
// decoder's slack at most ACE_MAX_AUDIO_SAMPLES, so T * ACE_LATENT_HOP and
// the buffer's int indexing hold -- neural-codec's bound, which a latent its
// --encode wrote past a run's most still meets (cadenza-audio #1266).
inline constexpr int ACE_MAX_DECODE_FRAMES = (ACE_MAX_AUDIO_SAMPLES - ACE_VAE_DEC_SLACK_SAMPLES) / ACE_LATENT_HOP;

static_assert(ACE_MAX_DECODE_FRAMES == 559238 && ACE_MAX_DECODE_FRAMES > ACE_MAX_LATENT_FRAMES &&
                  (long long) ACE_MAX_DECODE_FRAMES * ACE_LATENT_HOP + ACE_VAE_DEC_SLACK_SAMPLES <=
                      ACE_MAX_AUDIO_SAMPLES,
              "a decode's most frames, in samples with its slack, is a planar buffer's most");

// A latent frame's channels, and its bytes as f32: the layout every latent
// file and payload carries, [T, 64] time-major (cadenza-audio #1266).
inline constexpr int ACE_LATENT_CHANNELS    = 64;
inline constexpr int ACE_LATENT_FRAME_BYTES = ACE_LATENT_CHANNELS * (int) sizeof(float);

// The bound a latent is read to: a run's most -- every part ace-server takes
// -- or a decode's most, for neural-codec's own files.
enum class AceLatentBound { run, decode };

// What is wrong with a latent read from a file or the wire: nothing, no frame
// at all -- a zero-length decode, and a rate of inf or nan kbit/s -- bytes
// that are no whole number of frames, or more frames than its bound. One rule
// for ace-server's payloads and neural-codec's files (cadenza-audio #1266): a
// file's header that claims 1.2M frames is refused here, not after
// T * ACE_LATENT_HOP has overflowed an int.
enum class AceLatentFault { none, empty, ragged, past_run_most, past_decode_most };

inline constexpr AceLatentFault latent_frames_fault(long long frames, AceLatentBound bound) {
    if (frames < 1) {
        return AceLatentFault::empty;
    }
    if (bound == AceLatentBound::run) {
        return frames_past_run_most(frames) ? AceLatentFault::past_run_most : AceLatentFault::none;
    }
    return frames > ACE_MAX_DECODE_FRAMES ? AceLatentFault::past_decode_most : AceLatentFault::none;
}

// Whether a fault is one of too many frames: a 413 on the wire.
inline constexpr bool latent_fault_too_long(AceLatentFault fault) {
    return fault == AceLatentFault::past_run_most || fault == AceLatentFault::past_decode_most;
}

// A raw f32 latent of `bytes`, as its frames: the count, -1 with its fault.
struct AceLatentFrames {
    int            frames;
    AceLatentFault fault;
};

inline constexpr AceLatentFrames latent_payload_frames(unsigned long long bytes, AceLatentBound bound) {
    if (bytes % ACE_LATENT_FRAME_BYTES != 0) {
        return { -1, AceLatentFault::ragged };
    }
    const AceLatentFault fault = latent_frames_fault((long long) (bytes / ACE_LATENT_FRAME_BYTES), bound);
    return { fault == AceLatentFault::none ? (int) (bytes / ACE_LATENT_FRAME_BYTES) : -1, fault };
}

// A fault in words, for a tool's log: nullptr for none.
inline constexpr const char * latent_fault_words(AceLatentFault fault) {
    static_assert(ACE_MAX_LATENT_FRAMES == 15000 && ACE_MAX_DECODE_FRAMES == 559238, "the words below say the most");
    switch (fault) {
        case AceLatentFault::empty:
            return "no latent frame";
        case AceLatentFault::ragged:
            return "not a whole number of 64-channel f32 frames";
        case AceLatentFault::past_run_most:
            return "more latent frames than a run holds (15000, 600 s)";
        case AceLatentFault::past_decode_most:
            return "more latent frames than a decode holds (559238, 6.2 h)";
        case AceLatentFault::none:
            break;
    }
    return nullptr;
}

static_assert(latent_payload_frames(0, AceLatentBound::run).fault == AceLatentFault::empty &&
                  latent_payload_frames(ACE_LATENT_FRAME_BYTES, AceLatentBound::run).frames == 1 &&
                  latent_payload_frames(ACE_LATENT_FRAME_BYTES + 4, AceLatentBound::decode).fault ==
                      AceLatentFault::ragged &&
                  latent_payload_frames((unsigned long long) ACE_MAX_LATENT_FRAMES * ACE_LATENT_FRAME_BYTES,
                                        AceLatentBound::run)
                          .frames == ACE_MAX_LATENT_FRAMES &&
                  latent_payload_frames((unsigned long long) (ACE_MAX_LATENT_FRAMES + 1) * ACE_LATENT_FRAME_BYTES,
                                        AceLatentBound::run)
                          .fault == AceLatentFault::past_run_most &&
                  latent_frames_fault(ACE_MAX_LATENT_FRAMES + 1, AceLatentBound::decode) == AceLatentFault::none &&
                  latent_frames_fault(ACE_MAX_DECODE_FRAMES, AceLatentBound::decode) == AceLatentFault::none &&
                  latent_frames_fault(1200000, AceLatentBound::decode) == AceLatentFault::past_decode_most,
              "a latent of 1 to its bound's most frames, whole, is read; none, a part frame or one past it is not");

// The VAE tiles a tool may be asked for (--vae-chunk, --vae-overlap), in
// latent frames: a chunk of 1 to a run's most, an overlap of 0 to under half
// the chunk. A chunk of 0 is a stride of 0 the tiled VAE divides by, a
// negative one a run of no tile and no error, and one past ~1.1M frames
// overflows chunk * ACE_LATENT_HOP; a run's most frames is already one tile.
// An overlap of half the chunk or more leaves no core: the decoder halves it
// to fit, but the encoder halves it in samples and its tiles' rounded trims
// then cut an interior tile's core away -- --vae-chunk 4 --vae-overlap 3
// dropped most frames (cadenza-audio #1266).
inline constexpr int ACE_VAE_TILE_MOST = ACE_MAX_LATENT_FRAMES;

// Why a tool refuses these tile settings: nullptr when it takes them.
inline constexpr const char * vae_tiling_refused(int chunk, int overlap) {
    static_assert(ACE_VAE_TILE_MOST == 15000, "the refusals below say the most");
    if (chunk < 1 || chunk > ACE_VAE_TILE_MOST) {
        return "--vae-chunk must be 1 to 15000 latent frames";
    }
    if (overlap < 0 || 2LL * overlap >= chunk) {
        return "--vae-overlap must be 0 to under half --vae-chunk";
    }
    return nullptr;
}

static_assert((long long) ACE_VAE_TILE_MOST * ACE_LATENT_HOP <= std::numeric_limits<int>::max(),
              "the most tile, in samples, is an int");
static_assert(vae_tiling_refused(0, 16) && vae_tiling_refused(-1, 16) && vae_tiling_refused(ACE_VAE_TILE_MOST + 1, 0) &&
                  vae_tiling_refused(1024, -1) && vae_tiling_refused(4, 2) && vae_tiling_refused(4, 3) &&
                  vae_tiling_refused(256, 128) && !vae_tiling_refused(1, 0) && !vae_tiling_refused(5, 2) &&
                  !vae_tiling_refused(ACE_VAE_TILE_MOST, ACE_VAE_TILE_MOST / 2 - 1) && !vae_tiling_refused(1024, 0) &&
                  !vae_tiling_refused(1024, 16) && !vae_tiling_refused(1024, 64) && !vae_tiling_refused(256, 64) &&
                  !vae_tiling_refused(128, 16),
              "a chunk of 1 to the most, an overlap of 0 to under half of it; the defaults and benches' tiles");

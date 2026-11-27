// neural-codec.cpp: neural audio codec (Oobleck VAE)
//
// 2 modes:
//   encode: WAV -> latent file (.vae, .nac8, or .nac4)
//   decode: latent file -> WAV (48kHz stereo)
//
// Three latent formats, decode auto-detects:
//
//   .vae (default): flat [T, 64] f32, no header.
//     T = file_size / 256. 25Hz, ~6.4 KB/s, ~51 kbit/s.
//
//   .nac8 (--q8): symmetric per-frame int8 quantization.
//     header: "NAC8" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + int8[64] (64B) = 66B
//     25Hz, ~1.6 KB/s, ~13 kbit/s.
//
//   .nac4 (--q4): symmetric per-frame 4-bit quantization.
//     header: "NAC4" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + nibbles[32] (32B) = 34B
//     25Hz, ~850 B/s, ~6.8 kbit/s.
//
// Usage:
//   neural-codec --vae model.gguf --encode -i song.wav -o song.vae
//   neural-codec --vae model.gguf --encode --q8 -i song.wav -o song.nac8
//   neural-codec --vae model.gguf --encode --q4 -i song.wav -o song.nac4
//   neural-codec --vae model.gguf --decode -i song.nac4 -o song.wav

#include "audio-frames.h"
#include "audio-io.h"
#include "tool-main.h"
#include "vae-enc.h"
#include "vae.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// Q8 format constants
static const char NAC8_MAGIC[4] = { 'N', 'A', 'C', '8' };
static const int  NAC8_HEADER   = 8;   // 4B magic + 4B T_latent
static const int  NAC8_FRAME    = 66;  // 2B f16 scale + 64B int8

// Q4 format constants
static const char NAC4_MAGIC[4] = { 'N', 'A', 'C', '4' };
static const int  NAC4_HEADER   = 8;   // 4B magic + 4B T_latent
static const int  NAC4_FRAME    = 34;  // 2B f16 scale + 32B packed nibbles

static_assert(NAC8_FRAME == 2 + ACE_LATENT_CHANNELS && NAC4_FRAME == 2 + ACE_LATENT_CHANNELS / 2,
              "a quantized frame is its f16 scale and its 64 channels");

// A latent file's line in the log: its frames, length, size and rate. Only
// for a latent of at least one frame -- a read or an encode of none is
// refused first, so the rate is never inf or nan kbit/s (cadenza-audio #1266).
static void latent_log(const char * verb, const char * path, const char * kind, int T_latent, size_t bytes) {
    const float duration = latent_frames_sec(T_latent);
    fprintf(stderr, "[Latent] %s %s: %s, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n", verb, path, kind, T_latent,
            duration, (float) bytes / 1024.0f, (float) bytes * 8.0f / (duration * 1000.0f));
}

static void append_bytes(std::string & out, const void * p, size_t n) {
    out.append((const char *) p, n);
}

// A frame's largest magnitude: the range its symmetric quantization spans.
static float frame_amax(const float * frame) {
    float amax = 0.0f;
    for (int j = 0; j < ACE_LATENT_CHANNELS; j++) {
        float a = fabsf(frame[j]);
        if (a > amax) {
            amax = a;
        }
    }
    return amax;
}

// Q8 quantized latent: symmetric per-frame int8.
static std::string encode_latent_q8(const float * data, int T_latent) {
    std::string out;
    out.reserve(NAC8_HEADER + (size_t) T_latent * NAC8_FRAME);
    append_bytes(out, NAC8_MAGIC, 4);
    uint32_t t = (uint32_t) T_latent;
    append_bytes(out, &t, 4);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + (size_t) i * ACE_LATENT_CHANNELS;

        float       amax      = frame_amax(frame);
        float       scale     = amax / 127.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        append_bytes(out, &scale_f16, 2);

        // quantize
        int8_t q[ACE_LATENT_CHANNELS];
        float  inv = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < ACE_LATENT_CHANNELS; j++) {
            int v = (int) roundf(frame[j] * inv);
            q[j]  = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
        }
        append_bytes(out, q, sizeof(q));
    }
    return out;
}

// Q4 quantized latent
// Symmetric 4-bit: range [-7, 7], scale = amax / 7.0
// Packing: byte = (low & 0x0F) | (high << 4), two signed nibbles per byte
static std::string encode_latent_q4(const float * data, int T_latent) {
    std::string out;
    out.reserve(NAC4_HEADER + (size_t) T_latent * NAC4_FRAME);
    append_bytes(out, NAC4_MAGIC, 4);
    uint32_t t = (uint32_t) T_latent;
    append_bytes(out, &t, 4);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + (size_t) i * ACE_LATENT_CHANNELS;

        float       amax      = frame_amax(frame);
        float       scale     = amax / 7.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        append_bytes(out, &scale_f16, 2);

        // quantize and pack pairs into bytes
        float   inv = (scale > 0.0f) ? 7.0f / amax : 0.0f;
        uint8_t packed[ACE_LATENT_CHANNELS / 2];
        for (int j = 0; j < ACE_LATENT_CHANNELS / 2; j++) {
            int lo    = (int) roundf(frame[j * 2 + 0] * inv);
            int hi    = (int) roundf(frame[j * 2 + 1] * inv);
            lo        = lo < -7 ? -7 : (lo > 7 ? 7 : lo);
            hi        = hi < -7 ? -7 : (hi > 7 ? 7 : hi);
            packed[j] = (uint8_t) ((lo & 0x0F) | (hi << 4));
        }
        append_bytes(out, packed, sizeof(packed));
    }
    return out;
}

// Write a latent of at least one frame as f32 (quant 0, no header), Q8 or
// Q4: whether all of it reached the file (cadenza-audio #1266).
static bool write_latent(const char * path, const float * data, int T_latent, int quant) {
    // f32 is the latent's own bytes, written as they are; Q8 and Q4 build theirs.
    const std::string quantized = quant == 8 ? encode_latent_q8(data, T_latent) :
                                  quant == 4 ? encode_latent_q4(data, T_latent) :
                                               std::string();
    const void *      bytes     = quant ? (const void *) quantized.data() : (const void *) data;
    const size_t      size      = quant ? quantized.size() : (size_t) T_latent * ACE_LATENT_FRAME_BYTES;
    if (!audio_io_save_file(path, bytes, size)) {
        return false;
    }
    latent_log("Wrote", path, quant == 8 ? "Q8" : quant == 4 ? "Q4" : "f32", T_latent, size);
    return true;
}

// A latent's frame count refused, said why: true when it is (cadenza-audio
// #1266). A quantized file's header is asked before its size is, so a count
// past a run's most is refused before anything is sized by it.
static bool latent_refused(const char * path, AceLatentFault fault) {
    if (const char * why = latent_fault_words(fault)) {
        fprintf(stderr, "[Latent] Refused %s: %s\n", path, why);
        return true;
    }
    return false;
}

// Read a Q8 or Q4 latent's frames, its magic already read: [T_latent, 64] f32,
// dequantized. Caller frees.
static float * read_latent_quant(FILE * f, long fsize, const char * path, int quant, int * T_latent) {
    const int    header = quant == 8 ? NAC8_HEADER : NAC4_HEADER;
    const int    frame  = quant == 8 ? NAC8_FRAME : NAC4_FRAME;
    const char * kind   = quant == 8 ? "Q8" : "Q4";

    uint32_t t;
    if (fread(&t, 4, 1, f) != 1) {
        fprintf(stderr, "[Latent] %s: %s header cut short\n", path, kind);
        return NULL;
    }
    if (latent_refused(path, latent_frames_fault(t, AceLatentBound::decode))) {
        return NULL;
    }

    long expected = header + (long) t * frame;
    if (fsize != expected) {
        fprintf(stderr, "[Latent] %s size mismatch: expected %ld, got %ld\n", kind, expected, fsize);
        return NULL;
    }

    float * data = (float *) malloc((size_t) t * ACE_LATENT_FRAME_BYTES);
    if (!data) {
        fprintf(stderr, "[Latent] OOM allocating %s decode buffer for %u frames\n", kind, t);
        return NULL;
    }
    uint8_t q[ACE_LATENT_CHANNELS];  // a Q8 frame's 64 bytes, or a Q4 frame's 32
    for (int i = 0; i < (int) t; i++) {
        ggml_fp16_t scale_f16;
        if (fread(&scale_f16, 2, 1, f) != 1 || fread(q, 1, (size_t) frame - 2, f) != (size_t) frame - 2) {
            fprintf(stderr, "[Latent] %s: %s frame %d of %u cut short\n", path, kind, i, t);
            free(data);
            return NULL;
        }
        float   scale = ggml_fp16_to_fp32(scale_f16);
        float * out   = data + (size_t) i * ACE_LATENT_CHANNELS;
        if (quant == 8) {
            for (int j = 0; j < ACE_LATENT_CHANNELS; j++) {
                out[j] = (float) (int8_t) q[j] * scale;
            }
            continue;
        }
        // unpack signed nibbles
        for (int j = 0; j < ACE_LATENT_CHANNELS / 2; j++) {
            int lo = (int) (q[j] & 0x0F);
            int hi = (int) (q[j] >> 4);
            if (lo >= 8) {
                lo -= 16;
            }
            if (hi >= 8) {
                hi -= 16;
            }
            out[j * 2 + 0] = (float) lo * scale;
            out[j * 2 + 1] = (float) hi * scale;
        }
    }

    *T_latent = (int) t;
    latent_log("Read", path, kind, *T_latent, (size_t) fsize);
    return data;
}

// Read latent, auto-detect format (NAC8 -> Q8, NAC4 -> Q4, else f32).
// Returns [T_latent, 64] f32 (dequantized if quantized), 1 to a decode's most
// frames (ACE_MAX_DECODE_FRAMES: what T * ACE_LATENT_HOP and the audio
// buffer's int indexing hold) by the rule ace-server's latent payloads share
// -- not a run's most, as --encode writes longer latents than a run takes
// (cadenza-audio #1266) -- or NULL, said why. Caller frees.
static float * read_latent(const char * path, int * T_latent) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[Latent] Cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) {
        fprintf(stderr, "[Latent] Cannot size %s\n", path);
        fclose(f);
        return NULL;
    }

    // Check magic
    char magic[4] = {};
    if (fsize >= 8) {
        if (fread(magic, 1, 4, f) != 4) {
            fclose(f);
            return NULL;
        }
    }

    const int quant = memcmp(magic, NAC8_MAGIC, 4) == 0 ? 8 : (memcmp(magic, NAC4_MAGIC, 4) == 0 ? 4 : 0);
    if (quant) {
        float * data = read_latent_quant(f, fsize, path, quant, T_latent);
        fclose(f);
        return data;
    }

    // f32 format (no header, rewind)
    fseek(f, 0, SEEK_SET);
    const AceLatentFrames frames = latent_payload_frames((unsigned long long) fsize, AceLatentBound::decode);
    if (latent_refused(path, frames.fault)) {
        fclose(f);
        return NULL;
    }

    float * data = (float *) malloc(fsize);
    if (!data) {
        fprintf(stderr, "[Latent] OOM allocating f32 decode buffer (%ld bytes)\n", fsize);
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, fsize, f) != (size_t) fsize) {
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);

    *T_latent = frames.frames;
    latent_log("Read", path, "f32", *T_latent, (size_t) fsize);
    return data;
}

static void print_usage(const char * prog) {
    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]\n\n"
            "Required:\n"
            "  --vae <path>            VAE GGUF file\n"
            "  --encode | --decode     Encode audio to latent, or decode latent to WAV\n"
            "  -i <path>               Input (WAV/MP3 for encode, latent for decode)\n\n"
            "Output:\n"
            "  -o <path>               Output file (auto-named if omitted)\n"
            "  --q8                    Quantize latent to int8 (~13 kbit/s)\n"
            "  --q4                    Quantize latent to int4 (~6.8 kbit/s)\n"
            "  --format <fmt>          WAV format: wav16, wav24, wav32 (default: wav16)\n\n"
            "Output naming: song.wav -> song.vae (f32) or song.nac8 (Q8) or song.nac4 (Q4)\n"
            "               song.vae -> song.wav\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 1024)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
            "Latent formats (decode auto-detects):\n"
            "  .vae:  flat [T, 64] f32, no header. ~51 kbit/s.\n"
            "  .nac8: header + per-frame Q8. ~13 kbit/s.\n"
            "  .nac4: header + per-frame Q4. ~6.8 kbit/s.\n",
            prog);
}

static std::string auto_output(const char * input, const char * ext) {
    std::string s   = input;
    size_t      dot = s.rfind('.');
    if (dot != std::string::npos) {
        return s.substr(0, dot) + ext;
    }
    return s + ext;
}

static int codec_main(int argc, char ** argv) {
    const char * vae_path    = NULL;
    const char * input_path  = NULL;
    const char * output_path = NULL;
    int          chunk_size  = 1024;
    int          overlap     = 64;
    int          mode        = -1;  // 0 = encode, 1 = decode
    int          quant       = 0;   // 0 = f32, 8 = q8, 4 = q4
    WavFormat    wav_fmt     = WAV_S16;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae_path = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            bool dummy_mp3;
            if (!audio_parse_format(argv[++i], dummy_mp3, wav_fmt)) {
                fprintf(stderr, "Unknown format: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--vae-chunk") == 0 && i + 1 < argc) {
            // Any int: the tiling rule below says what a tile may be.
            if (!ace_tool_int_flag("VAE", "--vae-chunk", argv[++i], INT_MIN, INT_MAX, chunk_size)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--vae-overlap") == 0 && i + 1 < argc) {
            if (!ace_tool_int_flag("VAE", "--vae-overlap", argv[++i], INT_MIN, INT_MAX, overlap)) {
                return 1;
            }
        } else if (strcmp(argv[i], "--encode") == 0) {
            mode = 0;
        } else if (strcmp(argv[i], "--decode") == 0) {
            mode = 1;
        } else if (strcmp(argv[i], "--q8") == 0) {
            quant = 8;
        } else if (strcmp(argv[i], "--q4") == 0) {
            quant = 4;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!vae_path || !input_path || mode < 0) {
        print_usage(argv[0]);
        return 1;
    }
    // Tiles the VAE can run, refused before anything is read (cadenza-audio #1266).
    if (ace_tool_tiling_refused("VAE", chunk_size, overlap)) {
        return 1;
    }

    // Auto output names. f32 dumps land as .vae since they are the raw VAE
    // encoder output with no codec applied; nac8/nac4 keep their codec name.
    std::string out_str;
    if (!output_path) {
        if (mode == 0) {
            const char * ext = ".vae";
            if (quant == 8) {
                ext = ".nac8";
            }
            if (quant == 4) {
                ext = ".nac4";
            }
            out_str = auto_output(input_path, ext);
        } else {
            out_str = auto_output(input_path, ".wav");
        }
        output_path = out_str.c_str();
    }

    const char * quant_str = "";
    if (mode == 0 && quant == 8) {
        quant_str = " (Q8)";
    }
    if (mode == 0 && quant == 4) {
        quant_str = " (Q4)";
    }
    fprintf(stderr, "\n[VAE] Mode: %s%s\n", mode == 0 ? "encode" : "decode", quant_str);
    fprintf(stderr, "[VAE] Input:  %s\n", input_path);
    fprintf(stderr, "[VAE] Output: %s\n\n", output_path);

    // ENCODE
    if (mode == 0) {
        int     T_audio = 0;
        // Read to what decodes again: its whole frames a decode's most, by
        // the run rule's count (cadenza-audio #1266).
        float * planar  = audio_read_48k(input_path, &T_audio, ACE_MAX_DECODE_SOURCE_SAMPLES);
        if (!planar) {
            return 1;
        }

        float * audio = audio_planar_to_interleaved(planar, T_audio);
        free(planar);

        VAEEncoder enc = {};
        vae_enc_load(&enc, vae_path);

        int                max_T = (T_audio / ACE_LATENT_HOP) + ACE_VAE_ENC_SLACK_FRAMES;
        std::vector<float> latent((size_t) max_T * ACE_LATENT_CHANNELS);

        fprintf(stderr, "\n[VAE] Encoding %d samples (%.2fs)...\n", T_audio, samples_sec(T_audio));
        int T_latent = vae_enc_encode_tiled(&enc, audio, T_audio, latent.data(), max_T, chunk_size, overlap);
        free(audio);
        vae_enc_free(&enc);
        if (T_latent < 0) {
            return 1;
        }
        // An encode of no frame writes no file of nothing (cadenza-audio
        // #1266) -- held, though the VAE rounds a source of a sample up to a
        // frame today.
        if (T_latent == 0) {
            fprintf(stderr, "[VAE] FATAL: the encode made %s\n", latent_fault_words(AceLatentFault::empty));
            return 1;
        }
        if (frames_past_run_most(T_latent)) {
            fprintf(stderr, "[VAE] WARNING: %s; --decode takes it, ace-server refuses it\n",
                    latent_fault_words(AceLatentFault::past_run_most));
        }

        if (!write_latent(output_path, latent.data(), T_latent, quant)) {
            fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);
            return 1;
        }

        fprintf(stderr, "[VAE] Done.\n");
        return 0;
    }

    // DECODE (auto-detects f32 vs Q8 vs Q4 from file content)
    {
        int     T_latent = 0;
        float * latent   = read_latent(input_path, &T_latent);
        if (!latent) {
            return 1;
        }

        VAEGGML dec = {};
        vae_ggml_load(&dec, vae_path);

        int                max_T = T_latent * ACE_LATENT_HOP + ACE_VAE_DEC_SLACK_SAMPLES;
        std::vector<float> audio((size_t) 2 * max_T, 0.0f);

        fprintf(stderr, "\n[VAE] Decoding %d latent frames...\n", T_latent);
        int T_audio = vae_ggml_decode_tiled(&dec, latent, T_latent, audio.data(), max_T, chunk_size, overlap);
        free(latent);
        if (T_audio < 0) {
            vae_ggml_free(&dec);
            return 1;
        }

        vae_ggml_free(&dec);
        // A failed write fails the tool: no script reads success into no file
        // (cadenza-audio #1266).
        if (!audio_write(output_path, audio.data(), T_audio, ACE_SAMPLE_RATE, 0, wav_fmt)) {
            fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);
            return 1;
        }
        fprintf(stderr, "\n[VAE] Output: %s (%d samples, %.2fs @ %d Hz)\n", output_path, T_audio, samples_sec(T_audio),
                ACE_SAMPLE_RATE);
        fprintf(stderr, "[VAE] Done.\n");
        return 0;
    }
}

// A thrown fatal fails the tool with its status, as for every CLI
// (tool-main.h, cadenza-audio #1033, #1049).
int main(int argc, char ** argv) {
    return ace_tool_main(codec_main, argc, argv);
}

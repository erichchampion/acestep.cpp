// make-tool-inputs: the files the tools' refusal tests read (cadenza-audio
// #1266), written into a directory before them -- so the tests need no
// committed binaries, and each file's bytes are said here.
//
//   ./make-tool-inputs <dir>
//
// past-most.nac8  a Q8 header claiming 1.2M frames, and no frame after it
// empty.nac4      a Q4 header of no frame
// empty.vae       an f32 latent of no byte
// ragged.vae      an f32 latent of a frame and a part frame
// one-frame.vae   an f32 latent of one silent frame (25 ms of nothing)
// rate-1hz.wav    a WAV whose header claims 1 Hz
// one-second.wav  a WAV of 1 s of a 440 Hz tone at 48 kHz

#include "audio-frames.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static std::string g_dir;

static bool save(const char * name, const std::string & bytes) {
    const std::string path = g_dir + "/" + name;
    FILE *            f    = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "make-tool-inputs: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return fclose(f) == 0 && ok;
}

static void put_u16(std::string & out, uint16_t v) {
    out += (char) (v & 0xff);
    out += (char) (v >> 8);
}

static void put_u32(std::string & out, uint32_t v) {
    put_u16(out, (uint16_t) (v & 0xffff));
    put_u16(out, (uint16_t) (v >> 16));
}

// A 16-bit stereo PCM WAV of `samples` per channel at `rate`, a tone of `hz`.
static std::string wav(uint32_t rate, int samples, float hz) {
    std::string out = "RIFF";
    put_u32(out, 36 + (uint32_t) samples * 4);
    out += "WAVEfmt ";
    put_u32(out, 16);
    put_u16(out, 1);  // PCM
    put_u16(out, 2);  // stereo
    put_u32(out, rate);
    put_u32(out, rate * 4);
    put_u16(out, 4);
    put_u16(out, 16);
    out += "data";
    put_u32(out, (uint32_t) samples * 4);
    for (int t = 0; t < samples; t++) {
        const int16_t v = (int16_t) (8000.0f * sinf(2.0f * 3.14159265f * hz * (float) t / (float) rate));
        put_u16(out, (uint16_t) v);
        put_u16(out, (uint16_t) v);
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <dir>\n", argv[0]);
        return 2;
    }
    g_dir = argv[1];

    std::string past_most = "NAC8";
    put_u32(past_most, 1200000);
    std::string empty_q4 = "NAC4";
    put_u32(empty_q4, 0);

    const bool ok = save("past-most.nac8", past_most) && save("empty.nac4", empty_q4) && save("empty.vae", "") &&
                    save("ragged.vae", std::string((size_t) ACE_LATENT_FRAME_BYTES + 4, '\0')) &&
                    save("one-frame.vae", std::string((size_t) ACE_LATENT_FRAME_BYTES, '\0')) &&
                    save("rate-1hz.wav", wav(1, 100, 0.0f)) &&
                    save("one-second.wav", wav(ACE_SAMPLE_RATE, ACE_SAMPLE_RATE, 440.0f));
    return ok ? 0 : 1;
}

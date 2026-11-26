// ace-synth.cpp: ACE-Step synthesis CLI
// Thin wrapper: parses args, scans the model registry, calls pipeline-synth,
// writes output files. Model selection (synth_model, adapter, output_format)
// comes from the request JSON. The registry resolves names to GGUF paths
// under --models <dir> and --adapters <dir>.

#include "audio-frames.h"
#include "audio-io.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "request.h"
#include "synth-batch-runner.h"
#include "tool-main.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    AceSynthParams d;
    ace_synth_default_params(&d);

    fprintf(stderr, "acestep.cpp %s\n\n", ACE_VERSION);
    fprintf(stderr,
            "Usage: %s --models <dir> --request <json...> [options]\n\n"
            "Required:\n"
            "  --models <dir>          Directory of GGUF model files\n"
            "  --request <json...>     One or more request JSONs (from ace-lm --request)\n\n"
            "Optional:\n"
            "  --adapters <dir>        Directory of adapter files (enables JSON adapter field)\n"
            "  --src-audio <path>      Source audio (WAV or MP3)\n"
            "  --ref-audio <path>      Timbre reference audio (WAV or MP3)\n\n"
            "Memory control:\n"
            "  --vae-chunk <N>         Latent frames per tile (default: %d)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: %d)\n\n"
            "Debug:\n"
            "  --no-fa                 Disable flash attention\n"
            "  --no-batch-cfg          Split DiT CFG into two separate forwards\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --dump <dir>            Dump intermediate tensors\n",
            prog, d.vae_chunk, d.vae_overlap);
}

static int synth_main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    // Defaults live in ace_synth_default_params. CLI locals read from params
    // so there is exactly one place in the codebase that picks the numbers.
    AceSynthParams params;
    ace_synth_default_params(&params);

    std::vector<const char *> request_paths;
    const char *              models_dir     = NULL;
    const char *              adapters_dir   = NULL;
    const char *              src_audio_path = NULL;
    const char *              ref_audio_path = NULL;
    const char *              dump_dir       = NULL;
    bool                      use_fa         = true;
    bool                      use_batch_cfg  = true;
    bool                      clamp_fp16     = false;
    int                       vae_chunk      = params.vae_chunk;
    int                       vae_overlap    = params.vae_overlap;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--request")) {
            // Collect all following non-option args
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                request_paths.push_back(argv[++i]);
            }
        } else if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--adapters") && i + 1 < argc) {
            adapters_dir = argv[++i];
        } else if (!strcmp(argv[i], "--src-audio") && i + 1 < argc) {
            src_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--ref-audio") && i + 1 < argc) {
            ref_audio_path = argv[++i];
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (!strcmp(argv[i], "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(argv[i], "--no-batch-cfg")) {
            use_batch_cfg = false;
        } else if (!strcmp(argv[i], "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            vae_overlap = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }
    // Tiles the VAE can run, refused before any model loads (cadenza-audio #1266).
    if (ace_tool_tiling_refused("CLI", vae_chunk, vae_overlap)) {
        return 1;
    }
    if (request_paths.empty()) {
        fprintf(stderr, "[CLI] ERROR: --request required\n");
        usage(argv[0]);
        return 1;
    }

    // Parse all requests first: the first request drives model selection.
    int                      batch_n = (int) request_paths.size();
    std::vector<AceRequest>  reqs(batch_n);
    std::vector<std::string> basenames(batch_n);
    for (int ri = 0; ri < batch_n; ri++) {
        const char * rpath = request_paths[ri];
        if (!request_parse(&reqs[ri], rpath)) {
            fprintf(stderr, "[Ace-Synth] FATAL: failed to parse %s\n", rpath);
            return 1;
        }
        request_dump(&reqs[ri], stderr);
        // output basename: strip .json suffix
        basenames[ri] = rpath;
        size_t dot    = basenames[ri].rfind(".json");
        if (dot != std::string::npos) {
            basenames[ri] = basenames[ri].substr(0, dot);
        }
    }
    // The batch's takes, by the engine's one rule for which requests run
    // (ace_synth_batch_takes): computed once, asked of the refusal below and
    // expanded by after the load (cadenza-audio #1243, #1269).
    const AceSynthBatchTakes batch_takes = ace_synth_batch_takes(reqs.data(), batch_n);
    // The engine's batch rule, before any model is loaded: an unknown task,
    // requests that mix tasks, a request short of its caption (#1123), a
    // region bound or codes that are no such thing, a setting the batch
    // shares asked differently (#1159) -- of the requests that run.
    {
        AceSynthBatchRefusal why;
        if (ace_synth_batch_refused(reqs.data(), (int) batch_takes.takes.size(), &why)) {
            fprintf(stderr, "[Ace-Synth] FATAL: %s: %s\n", why.request >= 0 ? request_paths[why.request] : "the batch",
                    why.reason);
            return 1;
        }
    }
    fprintf(stderr, "[Ace-Synth] Batch: %d request(s)\n", batch_n);
    // A task that takes no source would drop one with a warning: say so now,
    // before any model loads, and leave --src-audio unread -- not checked,
    // not decoded, a whole song's worth of floats -- as the engine runs the
    // task without it (cadenza-audio #1157). ace-server does the same.
    if (src_audio_path && ace_synth_task_ignores_source(reqs[0].task_type.c_str())) {
        fprintf(stderr, "[Ace-Synth] WARNING: task '%s' ignores the given source: --src-audio not read\n",
                reqs[0].task_type.c_str());
        src_audio_path = NULL;
    }
    // A task that works from a source, given none: refused before any model
    // loads, by the run's own rule and reason (cadenza-audio #1160, #1193),
    // of the source as the run would get it (cadenza-audio #1194): with no
    // --src-audio, now; with one, once it is read, so a file that decodes to
    // nothing is refused as the run would refuse it, not after the load.
    auto inputs_refused = [&](const float * src, int len) {
        AceSynthBatchRefusal why;
        if (!ace_synth_inputs_refused(reqs[0].task_type.c_str(), src, len, NULL, 0, &why)) {
            return false;
        }
        fprintf(stderr, "[Ace-Synth] FATAL: task '%s': %s (--src-audio)\n", reqs[0].task_type.c_str(), why.reason);
        return true;
    };
    if (!src_audio_path && inputs_refused(NULL, 0)) {
        return 1;
    }

    // Scan the registry and resolve model paths from the first request.
    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Ace-Synth] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (adapters_dir) {
        registry_scan_adapters(&registry, adapters_dir);
    }
    if (registry.dit.empty() || registry.text_enc.empty() || registry.vae.empty()) {
        fprintf(stderr, "[Ace-Synth] FATAL: registry needs DiT, text-encoder and VAE models\n");
        return 1;
    }
    const ModelEntry * dit_entry =
        reqs[0].synth_model.empty() ? &registry.dit[0] : registry_find(registry.dit, reqs[0].synth_model.c_str());
    if (!dit_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: synth_model '%s' not found in registry\n", reqs[0].synth_model.c_str());
        return 1;
    }
    const ModelEntry * vae_entry =
        reqs[0].vae.empty() ? &registry.vae[0] : registry_find(registry.vae, reqs[0].vae.c_str());
    if (!vae_entry) {
        fprintf(stderr, "[Ace-Synth] FATAL: vae '%s' not found in registry\n", reqs[0].vae.c_str());
        return 1;
    }
    const AdapterEntry * adapter_entry = NULL;
    if (!reqs[0].adapter.empty()) {
        adapter_entry = registry_find_adapter(registry, reqs[0].adapter.c_str());
        if (!adapter_entry) {
            fprintf(stderr, "[Ace-Synth] FATAL: adapter '%s' not found (use --adapters <dir>)\n",
                    reqs[0].adapter.c_str());
            return 1;
        }
    }

    // Resolve output_format to (is_mp3, wav_fmt).
    bool      is_mp3  = true;
    WavFormat wav_fmt = WAV_S16;
    if (!audio_parse_format(reqs[0].output_format.c_str(), is_mp3, wav_fmt)) {
        fprintf(stderr, "[Ace-Synth] FATAL: invalid output_format '%s' (use: mp3, wav16, wav24, wav32)\n",
                reqs[0].output_format.c_str());
        return 1;
    }

    // Fill params from registry lookups and CLI flags.
    params.text_encoder_path = registry.text_enc[0].path.c_str();
    params.dit_path          = dit_entry->path.c_str();
    params.vae_path          = vae_entry->path.c_str();
    params.adapter_path      = adapter_entry ? adapter_entry->path.c_str() : NULL;
    params.adapter_scale     = reqs[0].adapter_scale;
    params.use_fa            = use_fa;
    params.use_batch_cfg     = use_batch_cfg;
    params.clamp_fp16        = clamp_fp16;
    params.vae_chunk         = vae_chunk;
    params.vae_overlap       = vae_overlap;
    params.dump_dir          = dump_dir;

    // Read source audio (cover/lego mode) and reference audio (timbre
    // conditioning) before any model loads, so a source that can't be read --
    // or one past what a run takes whole, which the engine would refuse
    // after its load -- costs no load (cadenza-audio #1266).
    float * src_interleaved = NULL;
    int     src_len         = 0;
    if (src_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(src_audio_path, &T_audio, ACE_MAX_RUN_SOURCE_SAMPLES);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --src-audio %s\n", src_audio_path);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Source audio: %.2fs @ %d Hz\n", samples_sec(T_audio), ACE_SAMPLE_RATE);

        src_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        // A conversion that could not allocate is said as that, not as the
        // input rule's "none was given" (cadenza-audio #1194). A read of no
        // sample, which malloc(0) may answer NULL, is the rule's to refuse.
        if (!src_interleaved && T_audio > 0) {
            fprintf(stderr, "[Ace-Synth] FATAL: out of memory converting --src-audio %s\n", src_audio_path);
            return 1;
        }
        src_len = T_audio;
        if (inputs_refused(src_interleaved, src_len)) {
            free(src_interleaved);
            return 1;
        }
    }

    float * ref_interleaved = NULL;
    int     ref_len         = 0;
    if (ref_audio_path) {
        int     T_audio = 0;
        float * planar  = audio_read_48k(ref_audio_path, &T_audio);
        if (!planar) {
            fprintf(stderr, "[Ace-Synth] FATAL: cannot read --ref-audio %s\n", ref_audio_path);
            free(src_interleaved);
            return 1;
        }
        fprintf(stderr, "[Ace-Synth] Reference audio: %.2fs @ %d Hz\n", samples_sec(T_audio), ACE_SAMPLE_RATE);
        ref_interleaved = audio_planar_to_interleaved(planar, T_audio);
        free(planar);
        ref_len = T_audio;
    }

    // Local store with the default STRICT policy: at most one GPU module
    // resident at a time for this one-shot CLI. No module sharing across runs,
    // so EVICT_STRICT frees the DiT before the VAE loads, and so on.
    ModelStore * store = store_create(EVICT_STRICT);
    AceSynth *   ctx   = ace_synth_load(store, &params);
    if (!ctx) {
        free(src_interleaved);
        free(ref_interleaved);
        store_free(store);
        return 1;
    }

    // Generate every request that runs in one DiT batch: each expanded to its
    // per-seed takes in groups[0], by the batch computed above -- clamped to
    // the DiT's kAceSynthMaxTakes, said with the requests it drops.
    std::vector<std::vector<AceRequest>> groups(1);
    std::vector<int>                     take_request, take_index;
    synth_batch_expand(reqs, batch_takes, groups[0], take_request, &take_index);
    for (const std::string & note :
         synth_batch_notes(batch_takes, batch_n, [&](int i) { return request_paths[(size_t) i]; })) {
        fprintf(stderr, "[Ace-Synth] %s\n", note.c_str());
    }
    const int                total_alloc = batch_takes.total;
    std::vector<AceAudio>    all_audio(total_alloc);
    std::vector<std::string> all_basenames(total_alloc);
    for (int b = 0; b < total_alloc; b++) {
        all_basenames[b] = basenames[(size_t) take_request[b]];
    }

    // Two-phase run: DiT resident for all groups, then VAE for all jobs.
    // The CLI does not expose latent IO yet: source and reference are always
    // audio, latent capture is disabled. The server reuses this runner with
    // the full feature.
    AceSynthBatchRefusal refused;
    const int rc = synth_batch_run(ctx, groups, src_interleaved, src_len, NULL, 0, ref_interleaved, ref_len, NULL, 0,
                                   all_audio.data(), nullptr, {}, &refused);
    if (rc != 0) {
        if (refused.reason) {
            fprintf(stderr, "[Ace-Synth] ERROR: the engine refused %s: %s\n",
                    refused.request >= 0 && refused.request < (int) all_basenames.size() ?
                        all_basenames[(size_t) refused.request].c_str() :
                        "the batch",
                    refused.reason);
        } else {
            // The engine's failure, the run's or its decode's, said as it
            // named it (cadenza-audio #1092, #1286).
            fprintf(stderr, "[Ace-Synth] ERROR: batch run failed: %s\n",
                    refused.failed ? refused.failed : "the engine's diagnostic is above");
        }
        for (auto & a : all_audio) {
            ace_audio_free(&a);
        }
        free(src_interleaved);
        free(ref_interleaved);
        ace_synth_free(ctx);
        store_free(store);
        return 1;
    }

    // Write output files. A take that could not be written fails the tool,
    // the others still written: no script reads success into a missing file
    // (cadenza-audio #1266).
    bool wrote_all = true;
    for (int b = 0; b < (int) all_audio.size(); b++) {
        if (!all_audio[b].samples) {
            continue;
        }
        const char * ext = is_mp3 ? ".mp3" : ".wav";
        char         out_path[1024];
        snprintf(out_path, sizeof(out_path), "%s%d%s", all_basenames[b].c_str(), take_index[b], ext);
        if (!audio_write(out_path, all_audio[b].samples, all_audio[b].n_samples, ACE_SAMPLE_RATE,
                         groups[0][b].mp3_bitrate, wav_fmt)) {
            fprintf(stderr, "[Ace-Synth Batch%d] FATAL: failed to write %s\n", b, out_path);
            wrote_all = false;
        }
        ace_audio_free(&all_audio[b]);
    }

    free(src_interleaved);
    free(ref_interleaved);
    ace_synth_free(ctx);
    store_free(store);
    if (!wrote_all) {
        return 1;
    }
    fprintf(stderr, "[Ace-Synth] All done\n");
    return 0;
}

int main(int argc, char ** argv) {
    return ace_tool_main(synth_main, argc, argv);
}

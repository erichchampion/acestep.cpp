// backend-config.cpp: the one embedder configuration (backend-config.h), defined
// once for the program (cadenza-audio #1049). ggml-free, as its header is: the
// acestep-config object library, which acestep-core, neural-codec and the
// ggml-free mp3-codec each link.

#include "backend-config.h"

AceBackendConfig & ace_backend_config() {
    static AceBackendConfig cfg;
    return cfg;
}

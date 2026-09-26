// load a model once, then switch a control vector between llama_decode calls
// schedule: turns none -> +v -> -v -> none (memory cleared per turn), then one run that switches +v on mid-generation

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    printf("\nusage: %s -m model.gguf --cvec FILE [--scale S] [--decode-only] [--switch-at N] [-p PROMPT | -f FILE] [-n N] [-t N] [--logits-out PREFIX]\n\n", argv0);
}

struct run_result {
    std::vector<llama_token> tokens;
    std::vector<float> logits0; // logits after the prompt
};

static std::string first_diff(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return std::to_string(i + 1);
        }
    }
    if (a.size() != b.size()) {
        return std::to_string(n + 1);
    }
    return "none (identical, n=" + std::to_string(n) + ")";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string cvec_path;
    std::string prompt = "Hello my name is";
    std::string logits_out;
    float scale       = 1.0f;
    bool  decode_only = false;
    int   switch_at   = 32;
    int   n_predict   = 64;
    int   n_threads   = 4;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_val = i + 1 < argc;
        if (arg == "-m" && has_val) {
            model_path = argv[++i];
        } else if (arg == "--cvec" && has_val) {
            cvec_path = argv[++i];
        } else if (arg == "--scale" && has_val) {
            scale = std::stof(argv[++i]);
        } else if (arg == "--decode-only") {
            decode_only = true;
        } else if (arg == "--switch-at" && has_val) {
            switch_at = std::stoi(argv[++i]);
        } else if (arg == "-p" && has_val) {
            prompt = argv[++i];
        } else if (arg == "-f" && has_val) {
            std::ifstream f(argv[++i]);
            if (!f) {
                fprintf(stderr, "error: failed to read %s\n", argv[i]);
                return 1;
            }
            prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        } else if (arg == "-n" && has_val) {
            n_predict = std::stoi(argv[++i]);
        } else if (arg == "-t" && has_val) {
            n_threads = std::stoi(argv[++i]);
        } else if (arg == "--logits-out" && has_val) {
            logits_out = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || cvec_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    string_process_escapes(prompt);

    ggml_backend_load_all();

    const int64_t t_load_start = ggml_time_us();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    cparams.n_batch         = 2048;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create the llama_context\n");
        return 1;
    }

    printf("model loaded once: %.1f ms (n_embd = %d, n_layer = %d)\n", (ggml_time_us() - t_load_start) / 1000.0, n_embd, n_layer);

    const auto cvec = common_control_vector_load({ { scale, cvec_path } });
    if (cvec.n_embd != n_embd) {
        fprintf(stderr, "error: control vector n_embd = %d, model n_embd = %d\n", cvec.n_embd, n_embd);
        return 1;
    }

    // pad to all layers, llama_set_adapter_cvec keeps old data for layers past the end of the buffer
    std::vector<float> v_pos(cvec.data);
    v_pos.resize((size_t) n_embd * n_layer, 0.0f);
    std::vector<float> v_neg(v_pos);
    for (float & x : v_neg) {
        x = -x;
    }

    for (int il = 1; il < n_layer; il++) {
        double norm = 0.0;
        for (int j = 0; j < n_embd; j++) {
            norm += (double) v_pos[(size_t) n_embd * (il - 1) + j] * v_pos[(size_t) n_embd * (il - 1) + j];
        }
        if (norm > 0.0) {
            printf("cvec: %s x %.3f -> layer %d, |v| = %.3f\n", cvec_path.c_str(), scale, il, sqrt(norm));
        }
    }

    const std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt, true, true);
    printf("prompt: %zu tokens, n_predict = %d, decode_only = %d\n", prompt_tokens.size(), n_predict, decode_only);
    printf("  prompt tokens:");
    for (auto t : prompt_tokens) {
        printf(" %d", t);
    }
    printf("\n\n");

    const std::vector<float> * v_cur = nullptr;
    std::vector<double> t_set_ms;
    std::vector<double> t_decode_after_set_ms;
    std::vector<double> t_decode_ms;
    bool just_set = false;

    auto set_cvec = [&](const std::vector<float> * v, const char * name) {
        if (v == v_cur) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        const int32_t err = v ? llama_set_adapter_cvec(ctx, v->data(), v->size(), n_embd, 1, n_layer)
                              : llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0);
        const double t = (ggml_time_us() - t0) / 1000.0;
        if (err) {
            fprintf(stderr, "error: llama_set_adapter_cvec failed (%d)\n", err);
            exit(1);
        }
        t_set_ms.push_back(t);
        printf("  set_cvec(%s): %.3f ms\n", name, t);
        v_cur = v;
        just_set = true;
    };

    auto decode = [&](llama_batch batch, bool record) {
        const int64_t t0 = ggml_time_us();
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: llama_decode failed\n");
            exit(1);
        }
        const double t = (ggml_time_us() - t0) / 1000.0;
        if (record) {
            (just_set ? t_decode_after_set_ms : t_decode_ms).push_back(t);
        }
        just_set = false;
    };

    // v_prompt is used for the prompt; v_switch is set before decoding generated token number switch_n (1-based)
    auto generate = [&](const char * label, const std::vector<float> * v_prompt, const char * name_prompt,
                        int switch_n, const std::vector<float> * v_switch, const char * name_switch) {
        printf("%s\n", label);
        run_result res;

        llama_memory_clear(llama_get_memory(ctx), true);

        set_cvec(v_prompt, name_prompt);

        const int64_t t0 = ggml_time_us();
        std::vector<llama_token> inp = prompt_tokens;
        decode(llama_batch_get_one(inp.data(), inp.size()), false);
        const double t_prompt = (ggml_time_us() - t0) / 1000.0;

        const float * logits = llama_get_logits_ith(ctx, -1);
        res.logits0.assign(logits, logits + n_vocab);

        const int64_t t1 = ggml_time_us();
        for (int i = 0; i < n_predict; i++) {
            logits = llama_get_logits_ith(ctx, -1);
            llama_token tok = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
            res.tokens.push_back(tok);
            if (llama_vocab_is_eog(vocab, tok) || i == n_predict - 1) {
                break;
            }
            if (switch_n > 0 && i + 1 == switch_n) {
                set_cvec(v_switch, name_switch);
            }
            decode(llama_batch_get_one(&tok, 1), true);
        }
        const double t_gen = (ggml_time_us() - t1) / 1000.0;

        std::string text;
        printf("  tokens:");
        for (auto t : res.tokens) {
            printf(" %d", t);
            text += common_token_to_piece(ctx, t);
        }
        printf("\n  text: \"%s\"\n", text.c_str());
        printf("  n_gen = %zu, prompt = %.1f ms, gen = %.1f ms\n\n", res.tokens.size(), t_prompt, t_gen);
        return res;
    };

    // decode-only: prompt unsteered, vector set before the first generated token is decoded
    auto turn = [&](const char * label, const std::vector<float> * v, const char * name) {
        if (decode_only) {
            return generate(label, nullptr, "none", 1, v, name);
        }
        return generate(label, v, name, 0, nullptr, "");
    };

    std::vector<run_result> turns;
    turns.push_back(turn("turn 1: none", nullptr, "none"));
    turns.push_back(turn("turn 2: +v",   &v_pos,  "+v"));
    turns.push_back(turn("turn 3: -v",   &v_neg,  "-v"));
    turns.push_back(turn("turn 4: none", nullptr, "none"));

    char label[64];
    snprintf(label, sizeof(label), "switch run: none, +v from token %d", switch_at);
    const run_result sw = generate(label, nullptr, "none", switch_at, &v_pos, "+v");

    printf("first_diff(turn1, turn2) = %s\n", first_diff(turns[0].tokens, turns[1].tokens).c_str());
    printf("first_diff(turn1, turn3) = %s\n", first_diff(turns[0].tokens, turns[2].tokens).c_str());
    printf("first_diff(turn2, turn3) = %s\n", first_diff(turns[1].tokens, turns[2].tokens).c_str());
    printf("first_diff(turn1, turn4) = %s\n", first_diff(turns[0].tokens, turns[3].tokens).c_str());
    printf("first_diff(turn1, switch) = %s (switch before decoding token %d, first steered token is %d)\n",
            first_diff(turns[0].tokens, sw.tokens).c_str(), switch_at, switch_at + 1);

    auto stats = [](std::vector<double> v) {
        if (v.empty()) {
            return std::string("n/a");
        }
        std::sort(v.begin(), v.end());
        return string_format("median %.3f ms, max %.3f ms, n = %zu", v[v.size() / 2], v.back(), v.size());
    };
    printf("\nset_cvec calls: %s\n", stats(t_set_ms).c_str());
    printf("single-token decode right after set_cvec: %s\n", stats(t_decode_after_set_ms).c_str());
    printf("single-token decode otherwise: %s\n", stats(t_decode_ms).c_str());

    if (!logits_out.empty()) {
        for (size_t i = 0; i < turns.size(); i++) {
            const std::string fname = logits_out + ".turn" + std::to_string(i + 1) + ".f32";
            FILE * f = fopen(fname.c_str(), "wb");
            if (!f) {
                fprintf(stderr, "error: failed to open %s\n", fname.c_str());
                return 1;
            }
            fwrite(turns[i].logits0.data(), sizeof(float), turns[i].logits0.size(), f);
            fclose(f);
        }
        printf("wrote prompt logits to %s.turn{1..4}.f32\n", logits_out.c_str());
    }

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

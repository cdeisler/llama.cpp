// Include custom headers for common utilities and llama library
#include "common.h"
#include "llama.h"
#include "build-info.h"

// Include standard libraries
#include <vector>
#include <cstdio>
#include <chrono>

int main(int argc, char ** argv) {
    // Initialize GPT (Generative Pre-trained Transformer) parameters with default values
    gpt_params params;
    params.seed = 42;                   // Seed value for random number generation
    params.n_threads = 4;               // Number of threads to use
    params.repeat_last_n = 64;          // The number of last tokens to repeat
    params.prompt = "The quick brown fox"; // Initial input prompt

    // Parse command line arguments to possibly override default parameter values
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1; // Exit with error code if parsing fails
    }

    // Print build information to the error stream
    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    // Set number of predictions if it hasn't been set by the user
    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    // Get default parameters for llama context
    auto lparams = llama_context_default_params();
    lparams.n_ctx     = params.n_ctx;
    lparams.seed      = params.seed;
    lparams.f16_kv    = params.memory_f16;
    lparams.use_mmap  = params.use_mmap;
    lparams.use_mlock = params.use_mlock;

    // Initialize variables
    auto n_past = 0;
    auto last_n_tokens_data = std::vector<llama_token>(params.repeat_last_n, 0);

    // Initialize llama context from model file
    auto ctx = llama_init_from_file(params.model.c_str(), lparams);

    // Tokenize the initial prompt
    auto tokens = std::vector<llama_token>(params.n_ctx);
    auto n_prompt_tokens = llama_tokenize(ctx, params.prompt.c_str(), tokens.data(), int(tokens.size()), true);

    // Exit if tokenization fails
    if (n_prompt_tokens < 1) {
        fprintf(stderr, "%s : failed to tokenize prompt\n", __func__);
        return 1;
    }

    // Evaluate the tokenized prompt
    llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, params.n_threads);

    // Store the evaluated tokens
    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    // Get the size of the internal state of the model
    const size_t state_size = llama_get_state_size(ctx);
    uint8_t * state_mem = new uint8_t[state_size];

    // Save internal state to a binary file
    {
        FILE *fp_write = fopen("dump_state.bin", "wb");
        llama_copy_state_data(ctx, state_mem);
        fwrite(state_mem, 1, state_size, fp_write);
        fclose(fp_write);
    }

    // Save tokens for later use
    const auto last_n_tokens_data_saved = std::vector<llama_token>(last_n_tokens_data);
    const auto n_past_saved = n_past;

    // First prediction run
    printf("\n%s", params.prompt.c_str());
    for (auto i = 0; i < params.n_predict; i++) {
        // Get the predicted logits and vocab size
        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);

        // Populate the candidate list
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

        // Sample the next token from the candidates
        auto next_token = llama_sample_token(ctx, &candidates_p);
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);

        // Print and evaluate the predicted token
        printf("%s", next_token_str);
        if (llama_eval(ctx, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    // Free memory allocated for the first context
    llama_free(ctx);

    // Initialize a second llama context
    auto ctx2 = llama_init_from_file(params.model.c_str(), lparams);

    // Load the previously saved state from the file
    {
        FILE *fp_read = fopen("dump_state.bin", "rb");
        if (state_size != llama_get_state_size(ctx2)) {
            fprintf(stderr, "\n%s : failed to validate state size\n", __func__);
            return 1;
        }
        const size_t ret = fread(state_mem, 1, state_size, fp_read);
        if (ret != state_size) {
            fprintf(stderr, "\n%s : failed to read state\n", __func__);
            return 1;
        }
        llama_set_state_data(ctx2, state_mem);
        fclose(fp_read);
    }

    delete[] state_mem;

    // Restore tokens for second run
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // Second prediction run with loaded state
    for (auto i = 0; i < params.n_predict; i++) {
        // Similar to the first run
        auto logits = llama_get_logits(ctx2);
        auto n_vocab = llama_n_vocab(ctx2);
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
        llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
        auto next_token = llama_sample_token(ctx2, &candidates_p);
        auto next_token_str = llama_token_to_str(ctx2, next_token);
        last_n_tokens_data.push_back(next_token);

        printf("%s", next_token_str);
        if (llama_eval(ctx2, &next_token, 1, n_past, params.n_threads)) {
            fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
            return 1;
        }
        n_past += 1;
    }

    printf("\n\n");

    // Exit normally
    return 0;
}

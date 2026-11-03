#include "llama.h"
#include "common.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Metadata only: no context, compute graph, weight mapping or inference.
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: native-prefix FIRST_GGUF PROMPT_AUDIT_JSON");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open audit JSON");
        const auto data = nlohmann::json::parse(input);
        const auto & agents = data.at("agents");
        if (!agents.is_array() || agents.empty() || agents.size() > 600) {
            throw std::runtime_error("expected 1..600 full agent prompts");
        }
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], params);
        if (!model) throw std::runtime_error("vocabulary load failed");
        const auto * vocab = llama_model_get_vocab(model);
        auto templates = common_chat_templates_init(model, "");
        nlohmann::json output = {{"scope", "native vocabulary and native Jinja; no model context or inference"},
            {"vocab_only", true}, {"context_created", false}, {"add_special", true}, {"parse_special", true},
            {"template", common_chat_templates_source(templates.get())}, {"agents", nlohmann::json::array()}};
        for (const auto & agent : agents) {
            common_chat_templates_inputs request;
            request.add_generation_prompt = true;
            request.use_jinja = true;
            request.chat_template_kwargs["reasoning_effort"] = "\"low\"";
            for (const auto & message : agent.at("messages")) {
                common_chat_msg row;
                row.role = message.at("role").get<std::string>();
                row.content = message.at("content").get<std::string>();
                request.messages.push_back(std::move(row));
            }
            const auto rendered = common_chat_templates_apply(templates.get(), request).prompt.str();
            const auto ids = common_tokenize(vocab, rendered, true, true);
            output["agents"].push_back({{"agent_id", agent.at("agent_id")},
                {"rendered_prompt", rendered}, {"token_ids", ids}});
        }
        templates.reset();
        llama_model_free(model);
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

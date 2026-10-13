#include "mission_interpreter.hpp"

#include <stdexcept>

#include <llama.h>
#include <yaml-cpp/yaml.h>

namespace nlp {

Settings load_settings(const std::string& yaml_path) {
    YAML::Node root;
    try{
        root = YAML::LoadFile(yaml_path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load YAML file: " + std::string(e.what()));
    }

    Settings s;
    s.model_path = root["model_path"].as<std::string>(s.model_path);
    s.grammar = root["grammar"].as<std::string>(s.grammar);
    s.context = root["context"].as<std::string>(s.context);
    s.n_ctx = root["n_ctx"].as<int>(s.n_ctx);
    s.n_gpu_layers = root["n_gpu_layers"].as<int>(s.n_gpu_layers);
    s.max_tokens = root["max_tokens"].as<int>(s.max_tokens);

     for (const auto& ex : root["examples"])
        s.examples.push_back({ex["command"].as<std::string>(), ex["output"].as<std::string>()});
 
    if (s.model_path.empty()) throw std::runtime_error(yaml_path + ": 'model_path' is required");
    if (s.grammar.empty())    throw std::runtime_error(yaml_path + ": 'grammar' is required");
    return s;
    }

MissionInterpreter::MissionInterpreter(Settings settings) : settings_(std::move(settings)) {
    system_prompt_ =
        "Decompose the user's command into an ordered list of (action, target) pairs.\n"
        "- action: the verb, lowercase.\n"
        "- target: what the action applies to, short, lowercase, keeping identifiers.\n"
        "- One pair per action, in execution order (in \"do X after Y\", Y comes first).\n"
        "- Do not add actions that are not in the command.";
    if (!settings_.context.empty()) system_prompt_ += "\nContext: " + settings_.context;
    
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = settings_.n_gpu_layers;
    model_ = llama_model_load_from_file(settings_.model_path.c_str(), mp);
    if (!model_) throw std::runtime_error("Model load failed: " + settings_.model_path);
    vocab_ = llama_model_get_vocab(model_);

    if(llama_sampler* g = llama_sampler_init_grammar(vocab_, settings_.grammar.c_str(), "root")) {
        llama_sampler_free(g);
    } else {
        llama_model_free(model_);
        throw std::runtime_error("Invalid configuration for grammar GBNF");
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = settings_.n_ctx;
    cp.n_batch = settings_.n_ctx;
    context_ = llama_init_from_model(model_, cp);
    if (!context_) {
        llama_model_free(model_);
        throw std::runtime_error("Failed to create context for model");
    }
}

MissionInterpreter::~MissionInterpreter() {
    if (context_) llama_free(context_);
    if (model_) llama_model_free(model_);
}

std::string MissionInterpreter::build_prompt(const std::string& command) const {
    std::vector<llama_chat_message> msgs = {{"system", system_prompt_.c_str()}};
    for (const Example& ex : settings_.examples) {
        msgs.push_back({"user", ex.command.c_str()});
        msgs.push_back({"assistant", ex.output.c_str()});
    }
    msgs.push_back({"user", command.c_str()});
 
    const char* tmpl = llama_model_chat_template(model_, nullptr);
    std::vector<char> buf(8192);
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    if (n > static_cast<int>(buf.size())) {
        buf.resize(n);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), buf.size());
    }
    if (n < 0) throw std::runtime_error("Template de chat não suportado por este modelo");
    return std::string(buf.data(), n);
}
 
std::string MissionInterpreter::interpret(const std::string& command) {
    const std::string prompt = build_prompt(command);
    llama_memory_clear(llama_get_memory(context_), true);  
 
    const int n_prompt = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    llama_tokenize(vocab_, prompt.c_str(), prompt.size(), tokens.data(), n_prompt, true, true);
    if (n_prompt + settings_.max_tokens > settings_.n_ctx)
        throw std::runtime_error("Comando longo demais para o contexto configurado");
 
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab_, settings_.grammar.c_str(), "root"));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
 
    std::string out;
    bool finished = false;
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token tok;
    for (int i = 0; i < settings_.max_tokens; ++i) {
        if (llama_decode(context_, batch) != 0) break;
        tok = llama_sampler_sample(smpl, context_, -1);
        if (llama_vocab_is_eog(vocab_, tok)) { finished = true; break; }
 
        char piece[256];
        const int n = llama_token_to_piece(vocab_, tok, piece, sizeof(piece), 0, false);
        if (n > 0) out.append(piece, n);
        batch = llama_batch_get_one(&tok, 1);
    }
    llama_sampler_free(smpl);
    if (!finished) throw std::runtime_error("Geração interrompida antes do fim (aumente max_tokens)");
    return out;
}
}// namespace nlp
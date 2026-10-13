#pragma once

#include <stdexcept>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace nlp{

struct Example{
	std::string command;
	std::string output;
};

struct Settings{
	std::string model_path;
	std::string grammar;
	std::string context;
	std::vector<Example> examples;
	int n_ctx = 2048;
	int n_gpu_layers = 99;
	int max_tokens = 512;
};

Settings load_settings(const std::string& yaml_path);

class MissionInterpreter{
public:
	explicit MissionInterpreter(Settings settings);
	~MissionInterpreter();

	MissionInterpreter(const MissionInterpreter&) = delete;
	MissionInterpreter& operator=(const MissionInterpreter&) = delete;

	// Retorna a decomposição do comando como JSON, no formato da gramática.
	std::string interpret(const std::string& command);

private:
	std::string build_prompt(const std::string& command) const;

	Settings settings_;
	std::string system_prompt_;
	llama_model* model_=nullptr;
	llama_context* context_=nullptr;
	const llama_vocab* vocab_=nullptr;
};

} // namespace nlp
// main.cpp — exemplo de uso
//   ./mission_cli config/config.yaml "comando"
//   ./mission_cli config/config.yaml --batch dataset.json resultados.jsonl
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "mission_interpreter.hpp"

// Escapa uma string como literal JSON.
static std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out + "\"";
}

// Roda cada caso de teste do dataset e grava uma linha JSON por caso:
//   {"id": ..., "predicted": <JSON do modelo ou null>, "latency_ms": ..., "error": <mensagem ou null>}
static int run_batch(nlp::MissionInterpreter& interpreter,
                     const std::string& dataset_path, const std::string& output_path) {
    const YAML::Node cases = YAML::LoadFile(dataset_path)["test_cases"];
    std::ofstream out(output_path);
    if (!out) throw std::runtime_error("Não foi possível criar " + output_path);

    int n_errors = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const std::string id = cases[i]["id"].as<std::string>();
        std::string predicted = "null", error = "null";

        const auto start = std::chrono::steady_clock::now();
        try {
            predicted = interpreter.interpret(cases[i]["command"].as<std::string>());
        } catch (const std::exception& e) {
            error = json_string(e.what());
            ++n_errors;
        }
        const double latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        out << "{\"id\": " << json_string(id) << ", \"predicted\": " << predicted
            << ", \"latency_ms\": " << latency_ms << ", \"error\": " << error << "}" << std::endl;
        std::cerr << "\r[" << i + 1 << "/" << cases.size() << "] " << id << std::flush;
    }
    std::cerr << "\nConcluído: " << cases.size() << " casos, " << n_errors << " erros -> "
              << output_path << "\n";
    return 0;
}

int main(int argc, char** argv) {
    const bool batch = argc == 5 && std::string(argv[2]) == "--batch";
    if (argc != 3 && !batch) {
        std::cerr << "uso: " << argv[0] << " <config.yaml> \"<comando>\"\n"
                  << "     " << argv[0] << " <config.yaml> --batch <dataset.json> <saida.jsonl>\n";
        return 1;
    }
    try {
        nlp::MissionInterpreter interpreter(nlp::load_settings(argv[1]));
        if (batch) return run_batch(interpreter, argv[3], argv[4]);
        std::cout << interpreter.interpret(argv[2]) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }
    return 0;
}

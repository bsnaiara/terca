#!/usr/bin/env python3
"""Compara a saída do mission_cli --batch com o esperado no dataset.

uso: python3 tools/evaluate.py <dataset.json> <resultados.jsonl> [metricas.json]

Imprime um resumo em Markdown e, se pedido, grava as métricas e os erros em JSON.
"""
import json
import statistics
import sys
from collections import Counter


def steps(decomposition):
    return [(t["action"], t["target"]) for t in decomposition["mission_decomposition"]]


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def evaluate(dataset_path, results_path):
    cases = {c["id"]: c for c in json.load(open(dataset_path))["test_cases"]}
    results = [json.loads(line) for line in open(results_path) if line.strip()]

    n = len(results)
    exact = same_count = wrong_order = 0
    aligned = action_ok = target_ok = pair_ok = 0
    tp = n_pred = n_gold = 0
    failures, mismatches = [], []
    action_confusion = Counter()

    for r in results:
        case = cases[r["id"]]
        gold = steps(case["expected"])
        n_gold += len(gold)
        if r["error"] is not None:
            failures.append({"id": r["id"], "command": case["command"], "error": r["error"]})
            continue

        pred = steps(r["predicted"])
        n_pred += len(pred)
        tp += sum((Counter(pred) & Counter(gold)).values())

        if pred == gold:
            exact += 1
        else:
            if Counter(pred) == Counter(gold):
                wrong_order += 1
            mismatches.append({"id": r["id"], "command": case["command"],
                               "expected": gold, "predicted": pred})
        if len(pred) == len(gold):
            same_count += 1

        for (pa, pt), (ga, gt) in zip(pred, gold):
            aligned += 1
            action_ok += pa == ga
            target_ok += pt == gt
            pair_ok += (pa, pt) == (ga, gt)
            if pa != ga:
                action_confusion[(ga, pa)] += 1

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [r["latency_ms"] for r in results]

    return {
        "cases": n,
        "dataset_cases": len(cases),
        "generation_errors": len(failures),
        "exact_match": exact / n,
        "step_count_match": same_count / n,
        "wrong_order_only": wrong_order / n,
        "action_accuracy": action_ok / aligned if aligned else 0.0,
        "target_accuracy": target_ok / aligned if aligned else 0.0,
        "pair_accuracy": pair_ok / aligned if aligned else 0.0,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies),
            "total_s": sum(latencies) / 1000,
        },
        "action_confusion": [{"expected": g, "predicted": p, "count": c}
                             for (g, p), c in action_confusion.most_common()],
        "failures": failures,
        "mismatches": mismatches,
    }


def print_summary(m):
    pct = lambda x: f"{100 * x:.1f}%"
    lat = m["latency_ms"]
    print(f"## Resultado ({m['cases']} de {m['dataset_cases']} casos)\n")
    print("| Métrica | Valor |\n|---|---|")
    print(f"| Acerto exato (decomposição inteira) | {pct(m['exact_match'])} |")
    print(f"| Número de passos correto | {pct(m['step_count_match'])} |")
    print(f"| Só a ordem errada | {pct(m['wrong_order_only'])} |")
    print(f"| Acurácia de ação (por posição) | {pct(m['action_accuracy'])} |")
    print(f"| Acurácia de alvo (por posição) | {pct(m['target_accuracy'])} |")
    print(f"| Precisão / revocação / F1 dos pares | {pct(m['pair_precision'])} / "
          f"{pct(m['pair_recall'])} / {pct(m['pair_f1'])} |")
    print(f"| Erros de geração | {m['generation_errors']} |")
    print(f"| Latência média / p50 / p95 / máx | {lat['mean']:.0f} / {lat['p50']:.0f} / "
          f"{lat['p95']:.0f} / {lat['max']:.0f} ms |")
    print(f"| Tempo total de inferência | {lat['total_s'] / 60:.1f} min |")

    if m["action_confusion"]:
        print("\n### Ações trocadas mais frequentes\n")
        for c in m["action_confusion"][:10]:
            print(f"- `{c['expected']}` → `{c['predicted']}`: {c['count']}")
    if m["mismatches"]:
        print(f"\n### Exemplos de erro ({len(m['mismatches'])} no total)\n")
        for x in m["mismatches"][:10]:
            print(f"- **{x['id']}**: {x['command']}\n  - esperado: {x['expected']}\n"
                  f"  - obtido:   {x['predicted']}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    metrics = evaluate(sys.argv[1], sys.argv[2])
    print_summary(metrics)
    if len(sys.argv) == 4:
        json.dump(metrics, open(sys.argv[3], "w"), ensure_ascii=False, indent=2)

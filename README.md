# DAMCS Reproduction: Multi-Agent Crafter with LLM Agents

[English](README.md) | [简体中文](README.zh-CN.md)

This repository is an executable reproduction and engineering adaptation of the DAMCS project: [LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://happyeureka.github.io/damcs/).

It runs multiple LLM-powered agents in a shared 2D Crafter world. The team must cooperate through resource gathering, crafting, memory, and structured information sharing to obtain a diamond.

> This repository is based on the original project at [13RENDA/Mcrafter_LLM_Agent](https://github.com/13RENDA/Mcrafter_LLM_Agent). Please cite the DAMCS paper and respect the licenses of the upstream project and its dependencies when using this code.

## What is reproduced

The reproduction includes the following DAMCS components:

- **Multi-Agent Crafter (MAC)**: a shared, partially observable Crafter environment with configurable agent counts.
- **LLM agents**: agents generate structured plans and actions from their observations and memory.
- **Adaptive knowledge-graph memory**: agents maintain episodic, semantic, procedural, and long-term memory.
- **Structured cooperation**: agents coordinate resource collection, crafting, and resource sharing.
- **Experiment tooling**: CSV statistics, visual observations, multi-view GUI, plots, and optional video generation.

The default task is to collect a diamond. This requires the usual dependency chain: wood, workbench, wooden pickaxe, stone, stone pickaxe, coal/iron, furnace, iron pickaxe, and finally diamond.

## Project layout

```text
.
|-- crafter/                   # Multi-Agent Crafter environment and rendering
|-- memory_system/             # Agent, LLM integration, memory, knowledge graph, planning
|-- GUI/                       # Standalone HumanAgent GUI variants
|-- scripts/                   # Environment / LLM smoke checks and Qwen helper scripts
|-- descriptions/              # Prompts and environment descriptions
|-- icons/                     # Plot and GUI assets
|-- run.py                     # Main headless experiment entry point
|-- run_gui_multi_view.py      # Multi-view interactive GUI entry point
|-- plot.py                    # Experiment-statistics plotting helpers
|-- make_video_from_results.py # Video generation from saved agent frames
|-- requirements.txt           # Windows-friendly Python dependencies
|-- environment-win.yml        # Recommended minimal Conda environment definition
`-- environment.yml            # Full exported Conda environment specification
```

## Requirements

- Windows, Linux, or macOS with Conda (tested workflow uses Windows)
- Python 3.10
- An LLM provider account: Azure OpenAI, OpenAI, or Qwen/DashScope
- A display environment for GUI mode (`pygame`)

## Installation

The recommended reproducible setup is the small, platform-friendly environment definition:

```powershell
conda env create -f environment-win.yml
conda activate mcrafter
```

Alternatively, use the full exported environment when it matches your platform:

```powershell
conda env create -f environment.yml
conda activate mcrafter
```

## Configure an LLM provider

Copy the template and set one provider's credentials. Never commit `.env` or API-key files.

```powershell
Copy-Item .env.example .env
```

The runtime reads environment variables. If your shell does not automatically load `.env`, set the variables in the shell/session that starts Python.

### Qwen / DashScope example

```dotenv
LLM_PROVIDER=qwen
QWEN_API_KEY=your-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.7-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

Azure OpenAI and OpenAI examples are provided in [.env.example](.env.example).

Validate the provider configuration before launching a full experiment:

```powershell
python scripts\check_llm_config.py
python scripts\smoke_llm.py --mode both
```

`smoke_llm.py` makes real provider requests when configuration is present, so it may incur API usage.

## Run the reproduction

Start with a short run. `--results_dir` is an output root; the runner creates an
isolated UTC/run-id subdirectory automatically:

```powershell
python run.py --agent_num 1 --step_num 10 --results_dir results\smoke --no_plot
```

Run the main six-agent DAMCS-style experiment:

```powershell
python run.py --agent_num 6 --step_num 350 --results_dir results\damcs_6_agents
```

This is the formal reproduction command used for a normal headless experiment. It requires a configured LLM provider because the agents call the selected model during planning and communication. For a quick demonstration or debugging run, reduce `--step_num` and use `--no_plot`.

The headless runner uses LangGraph as its default orchestration backend. This is
a compatibility-first migration: the existing agents, prompts, parallel LLM
processor, action ordering, and `Env.step()` behavior are unchanged. To compare
against the original Python `while` loop, select the legacy backend explicitly:

```powershell
python run.py --orchestrator legacy --agent_num 1 --step_num 10 --no_plot
```

The LangGraph flow is implemented in `langgraph_simulation.py` as:

```text
initialize -> prepare_actions -> step_environment
           -> update_agent_states
           -> Send(agent planner subgraph) x N
           -> reducer -> apply_agent_responses
           -> complete_step -> repeat/end
```

Each AI agent planner is a LangGraph subgraph with model-call and response
validation nodes. Planner branches use LangGraph fan-out/fan-in rather than the
legacy `ThreadPoolExecutor`; `--max_concurrency` limits simultaneous branches.

LangGraph runs commit an atomic snapshot after every complete environment step.
The snapshot contains the Crafter environment, all agents and memories, graph
state, and Python/NumPy random states. The run id is printed at startup. Resume
an interrupted run with the same simulation parameters:

```powershell
python run.py --run_id damcs-study-01 --agent_num 6 --step_num 350

# After an interruption:
python run.py --run_id damcs-study-01 --agent_num 6 --step_num 350 --resume
```

Checkpoints are stored in `checkpoints/damcs.sqlite` by default; use
`--checkpoint_db` to select another database. Successful per-agent LLM results
are cached by run, step, agent, and request hash, so recovery does not repeat a
completed model request. A crash in the middle of a step resumes from the last
fully committed step.

Checkpoint snapshots use a versioned JSON data model. World materials, world
objects, players, agent memory objects, knowledge graphs, arrays, and random
states are encoded explicitly; live Python environment and Agent objects are no
longer pickled into the database. Resume with the same Python and project code.
Checkpoints created by the earlier pickle-based schema are intentionally rejected.

Useful command-line options:

| Option | Default | Description |
| --- | ---: | --- |
| `--agent_num` | `6` | Number of agents in the shared environment. |
| `--step_num` | `350` | Maximum number of environment steps per round. |
| `--num_of_turn` | `1` | Number of simulation rounds. |
| `--results_dir` | `results` | Root directory under which an isolated UTC/run-id experiment directory is created. |
| `--no_plot` | off | Skip the plot displayed after a run. Useful for headless runs. |
| `--human_agents` | empty | IDs excluded from LLM planning. Headless terminal input is not connected yet; use the GUI for manual control. |
| `--orchestrator` | `langgraph` | Use `langgraph` or the original `legacy` loop. |
| `--run_id` | generated | Stable identifier printed at startup and required by `--resume`. |
| `--resume` | off | Resume the latest complete step for `--run_id`. Requires one simulation round. |
| `--checkpoint_db` | `checkpoints/damcs.sqlite` | SQLite file containing runtime snapshots and LLM response cache. |
| `--max_concurrency` | `6` | Maximum simultaneous LangGraph Agent planner branches. |

Each new invocation creates a Windows-safe directory such as
`results/20260816T031522123456Z_damcs-study-01/`. Independent runs therefore do
not append to or overwrite one another. The layout is:

```text
results/<UTC>_<run_id>/
  manifest.json
  all_rounds_stats.csv
  result_fig.png                 # omitted with --no_plot
  round_001/
    stats.csv
    agent_0/                     # observation frames and knowledge graphs
    agent_1/
  round_002/                     # when --num_of_turn is greater than 1
```

`manifest.json` records the exact configuration, status, rounds, and resume
count. A resumed LangGraph run reads its original experiment directory from the
checkpoint database and reuses it, even if a different `--results_dir` is
supplied. The per-round and combined CSV files are rewritten from restored state
instead of appended, so resuming a completed run does not duplicate rows.

## Run the multi-view GUI

```powershell
python run_gui_multi_view.py --agent_num 4 --step_num 150 --fps 5
```

To control selected agents manually:

```powershell
python run_gui_multi_view.py --human_agents 0,2 --agent_num 4 --step_num 150
```

In the headless `run.py` path, `--human_agents` currently excludes those IDs
from LLM planning but does not invoke terminal input. Manual intervention is
available through the GUI path above; durable LangGraph human-in-the-loop review
is not implemented yet.

Controls:

- `v`: switch between single-agent, multi-agent, and overview modes
- `l`: switch multi-view layout (horizontal, vertical, grid)
- `0`-`9`: select the followed agent in single-view mode
- `h`: enter HumanAgent action input mode when HumanAgents are enabled
- `p`: pause or resume the simulation
- `Esc`: quit

## Relation to the original DAMCS project

This repository is a reproduction and engineering adaptation, not the official DAMCS release. It follows the same high-level idea from the paper and project page: decentralized LLM agents cooperate in Multi-Agent Crafter using structured memory and communication.

Main differences from the paper/original upstream project:

- The code is organized as a runnable local reproduction with additional setup, smoke-test, and provider-configuration scripts.
- The README documents a Windows/Conda workflow and Qwen/DashScope usage in addition to Azure OpenAI/OpenAI style configuration.
- Local API keys are expected to live in `.env`; secret files and generated outputs are ignored by Git.
- The default command is a practical DAMCS-style six-agent run, while exact paper metrics may require matching the paper's full experimental settings, model choices, random seeds, and evaluation protocol.
- Some GUI and helper scripts are retained for visualization and debugging, so this repository is closer to a reproducible course/project implementation than a minimal paper artifact.

## Verification without an LLM

Use the environment smoke test to verify Crafter, rendering dependencies, and multi-agent stepping without sending an API request:

```powershell
python scripts\smoke_env.py --agent_num 2 --step_num 5 --results_dir results\env_smoke
```

## Results, plots, and videos

The main runner produces CSV statistics in the generated experiment directory. It opens a comparison plot unless `--no_plot` is supplied.

To make a video from saved `agent_<id>` frame folders:

```powershell
python make_video_from_results.py --results_dir results\<UTC>_<run_id>\round_001 --output_dir videos --multi_panel --fps 5
```

`plot.py` contains reusable plotting functions. The main runner invokes `Plotter` automatically after a run unless disabled with `--no_plot`.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `LLMConfigurationError` or authentication failure | Check the provider variables in `.env` / the active shell, then run `python scripts\check_llm_config.py`. |
| GUI does not open | Activate `mcrafter`, install dependencies from `requirements.txt`, and run in a desktop session with `pygame` support. |
| Slow or costly runs | Test with `--agent_num 1 --step_num 10 --no_plot` first. LLM calls can take time and incur provider charges. |
| Cannot find a run's outputs | Read the `Experiment output:` path printed at startup, or locate `manifest.json` under `--results_dir`. |
| No model response | Check network access, API quota, selected model/deployment name, and provider timeout/retry settings. |

## Citation

```bibtex
@misc{yang2025damcs,
  title={LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning},
  author={Hanqing Yang and Jingdi Chen and Marie Siew and Tania Lorido-Botran and Carlee Joe-Wong},
  year={2025},
  eprint={2502.05453},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2502.05453}
}
```

See the [DAMCS project page](https://happyeureka.github.io/damcs/) and the [paper on arXiv](https://arxiv.org/abs/2502.05453) for the original method and evaluation.

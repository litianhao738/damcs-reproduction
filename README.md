# DAMCS Reproduction: Multi-Agent Crafter with LLM Agents

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
├── crafter/                 # Multi-Agent Crafter environment and rendering
├── memory_system/           # Agent, LLM integration, memory, knowledge graph, planning
├── GUI/                     # Standalone HumanAgent GUI variants
├── scripts/                 # Environment / LLM smoke checks and Qwen helper scripts
├── descriptions/            # Prompts and environment descriptions
├── icons/                   # Plot and GUI assets
├── run.py                   # Main headless experiment entry point
├── run_gui_multi_view.py    # Multi-view interactive GUI entry point
├── plot.py                  # Experiment-statistics plotting helpers
├── make_video_from_results.py # Video generation from saved agent frames
├── requirements.txt         # Windows-friendly Python dependencies
├── environment-win.yml      # Recommended minimal Conda environment definition
└── environment.yml          # Full exported Conda environment specification
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

Start with a short run and a dedicated output directory:

```powershell
python run.py --agent_num 1 --step_num 10 --results_dir results\smoke --no_plot
```

Run a standard six-agent DAMCS-style experiment:

```powershell
python run.py --agent_num 6 --step_num 350 --results_dir results\damcs_6_agents
```

Useful command-line options:

| Option | Default | Description |
| --- | ---: | --- |
| `--agent_num` | `6` | Number of agents in the shared environment. |
| `--step_num` | `350` | Maximum number of environment steps per round. |
| `--num_of_turn` | `1` | Number of simulation rounds. |
| `--results_dir` | `results` | Directory for CSV statistics and observations. Use a unique directory per experiment. |
| `--no_plot` | off | Skip the plot displayed after a run. Useful for headless runs. |
| `--human_agents` | empty | Comma-separated IDs of human-controlled agents, such as `0,2`. |

Each run writes per-round statistics and appends combined statistics to `all_rounds_stats.csv` within the chosen results directory. Use a new `--results_dir` for each experiment to avoid combining independent runs.

## Run the multi-view GUI

```powershell
python run_gui_multi_view.py --agent_num 4 --step_num 150 --fps 5
```

To control selected agents manually:

```powershell
python run_gui_multi_view.py --human_agents 0,2 --agent_num 4 --step_num 150
```

Controls:

- `v`: switch between single-agent, multi-agent, and overview modes
- `l`: switch multi-view layout (horizontal, vertical, grid)
- `0`–`9`: select the followed agent in single-view mode
- `h`: enter HumanAgent action input mode when HumanAgents are enabled
- `p`: pause or resume the simulation
- `Esc`: quit

## Verification without an LLM

Use the environment smoke test to verify Crafter, rendering dependencies, and multi-agent stepping without sending an API request:

```powershell
python scripts\smoke_env.py --agent_num 2 --step_num 5 --results_dir results\env_smoke
```

## Results, plots, and videos

The main runner produces CSV statistics in the supplied output directory. It opens a comparison plot unless `--no_plot` is supplied.

To make a video from saved `agent_<id>` frame folders:

```powershell
python make_video_from_results.py --results_dir results\your_run --output_dir videos --multi_panel --fps 5
```

`plot.py` contains reusable plotting functions. The main runner invokes `Plotter` automatically after a run unless disabled with `--no_plot`.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `LLMConfigurationError` or authentication failure | Check the provider variables in `.env` / the active shell, then run `python scripts\check_llm_config.py`. |
| GUI does not open | Activate `mcrafter`, install dependencies from `requirements.txt`, and run in a desktop session with `pygame` support. |
| Slow or costly runs | Test with `--agent_num 1 --step_num 10 --no_plot` first. LLM calls can take time and incur provider charges. |
| Mixed or unexpected CSV results | Use a fresh `--results_dir` for each experiment. |
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

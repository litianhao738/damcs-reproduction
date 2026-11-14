# DAMCS Reproduction: Multi-Agent Crafter with LLM Agents

This repository is an executable reproduction and engineering adaptation of the DAMCS project: [LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning](https://happyeureka.github.io/damcs/).

It runs multiple LLM-powered agents in a shared 2D Crafter world. The team must cooperate through resource gathering, crafting, memory, and structured information sharing to obtain a diamond.

> This repository is based on the original project at [13RENDA/Mcrafter_LLM_Agent](https://github.com/13RENDA/Mcrafter_LLM_Agent). Please cite the DAMCS paper and respect the licenses of the upstream project and its dependencies when using this code.

## What is reproduced

The reproduction includes the following DAMCS components:

- **Multi-Agent Crafter (MAC)**: a shared, partially observable Crafter environment with configurable agent counts.
- **LLM agents**: agents generate structured plans and actions from their observations and memory. Agents are built on the [AgentScope](https://github.com/agentscope-ai/agentscope) framework.
- **Adaptive knowledge-graph memory**: agents maintain episodic, semantic, procedural, and long-term memory.
- **Structured cooperation**: agents coordinate resource collection, crafting, and resource sharing.
- **Experiment tooling**: CSV statistics, visual observations, multi-view GUI, plots, and optional video generation.

The default task is to collect a diamond. This requires the usual dependency chain: wood, workbench, wooden pickaxe, stone, stone pickaxe, coal/iron, furnace, iron pickaxe, and finally diamond.

## Project layout

```text
.
|-- crafter/                   # Multi-Agent Crafter environment and rendering
|-- memory_system/             # AgentScope agent, LLM integration, memory, knowledge graph, planning
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

If `mcrafter` already exists, update its Python dependencies after pulling changes:

```powershell
conda run -n mcrafter python -m pip install -r requirements.txt
conda run -n mcrafter python -m pip check
conda run -n mcrafter python scripts\smoke_agentscope.py
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

If your Qwen credentials are in the local CSV supported by the Windows launcher, it loads them into the child process automatically:

```powershell
.\scripts\run_with_qwen.ps1 -PythonArgs @("scripts\smoke_llm.py", "--mode", "both", "--strict")
.\scripts\run_with_qwen.ps1 -PythonArgs @("run.py", "--agent_num", "2", "--step_num", "2", "--results_dir", "results\agentscope_smoke", "--no_plot")
```

The launcher runs from the project directory, streams progress, and returns Python's exit code.

## Run the reproduction

Start with a short run and a dedicated output directory:

```powershell
python run.py --agent_num 1 --step_num 10 --results_dir results\smoke --no_plot
```

Run the main six-agent DAMCS-style experiment:

```powershell
python run.py --agent_num 6 --step_num 350 --results_dir results\damcs_6_agents
```

This runs a headless experiment with cooperation checks and up to two targeted repair rounds enabled by default. Use `--coordination_mode off` for the original DAMCS-style planning baseline. A configured LLM provider is required because the agents call the selected model during planning and communication. For a quick demonstration or debugging run, reduce `--step_num` and use `--no_plot`.

Useful command-line options:

| Option | Default | Description |
| --- | ---: | --- |
| `--agent_num` | `6` | Number of agents in the shared environment. |
| `--step_num` | `350` | Maximum number of environment steps per round. |
| `--num_of_turn` | `1` | Number of simulation rounds. |
| `--coordination_mode` | `repair` | `off`: original planning; `diagnose`: report checks; `repair`: targeted replanning and holding unresolved immediate operations. |
| `--repair_rounds` | `2` | Maximum extra targeted replanning rounds per cycle: `0`, `1`, or `2`. |
| `--results_dir` | `results` | Directory for CSV statistics and observations. Use a unique directory per experiment. |
| `--save_global_maps` | off | Save complete world-map screenshots under `<results_dir>/global_maps`. |
| `--global_map_start` | `0` | First step to save when `--save_global_maps` is enabled. |
| `--global_map_unit` | `8` | Pixels per map cell in saved complete world-map screenshots. |
| `--no_plot` | off | Skip the plot displayed after a run. Useful for headless runs. |
| `--human_agents` | empty | Comma-separated IDs of human-controlled agents, such as `0,2`. |

Each run writes per-round statistics and appends combined statistics to `all_rounds_stats.csv` within the chosen results directory. Use a new `--results_dir` for each experiment to avoid combining independent runs.

Agent observations and knowledge graphs are also saved under `<results_dir>/agent_<id>`. Runs with multiple rounds put these agent folders under `<results_dir>/round_<n>`.

### Save the complete map near the final mining steps

The normal observations are each agent's local view plus inventory. To save the full 64×64 world map, including terrain, resources, players, and other objects, enable global-map capture. For the previously successful 150-step run, save from step 130 onward:

```powershell
python run.py --agent_num 6 --step_num 150 --num_of_turn 1 --results_dir results\qwen37_plus_20260526_agent_6a_150s_with_global_map --save_global_maps --global_map_start 130 --global_map_unit 8 --no_plot
```

The generated files are named `global_map_step_0130.png`, `global_map_step_0131.png`, and so on, under `<results_dir>/global_maps`. Because the LLM can produce nondeterministic responses, a rerun may complete at a different step; saving from step 130 captures the actual final phase of that rerun.

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
- `0`-`9`: select the followed agent in single-view mode
- `h`: enter HumanAgent action input mode when HumanAgents are enabled
- `p`: pause or resume the simulation
- `Esc`: quit

## AgentScope integration

The agent layer runs on [AgentScope](https://github.com/agentscope-ai/agentscope) 1.0:

- The tested dependency set pins `agentscope==1.0.21`, `mcp==1.30.0`, and `openai==1.109.1`. Older OpenAI SDK versions such as 1.66.3 lack the stable `chat.completions.parse` method used by AgentScope for structured output.
- `memory_system/agent.py`: `Agent` subclasses AgentScope's `AgentBase`. Its async `reply` is one planning step: it receives the step context as a `Msg` and returns a `Msg` whose `metadata` holds the structured plan (`ResponseEvent`).
- Both synchronous planning and the multi-agent scheduler validate the returned message's metadata, including any changes made by AgentScope reply hooks.
- `memory_system/llm_api.py`: chat and embedding calls go through AgentScope's `OpenAIChatModel` and `OpenAITextEmbedding` for all three providers (Azure OpenAI, OpenAI, Qwen/DashScope OpenAI-compatible endpoint). Prompts are AgentScope `Msg` objects with text and image blocks, formatted by `OpenAIChatFormatter`.
- `utils.py`: all agents that need a new plan in a step reply concurrently with `asyncio.gather`. The simulation loop and GUI stay synchronous and call into AgentScope through `run_async`.
- `run_async` rejects synchronous calls from its own event loop to prevent deadlocks. Async AgentScope code should use `await aget_completion(...)` directly. `LLM_MAX_RETRIES` must be at least 1.

Crafter, the DAMCS memory and knowledge-graph logic, and collaboration-context construction remain project components. AgentScope supplies the agent lifecycle, messages, formatters, and model adapters; provider environment variables remain the same.

## Cooperation checks and targeted repair

`memory_system/coordination.py` checks each agent's submitted plan against submitted observations and inventories. Recipes come directly from `crafter/data.yaml`. The current MA-Crafter implementation covers:

- Crafting materials and facilities in the actual 3x3 neighborhood, including diagonals.
- Placement materials, target terrain, and occupancy.
- Collection tool requirements and valid `do` targets.
- Share recipients and stock, with valid planned transfers applied in execution order before checking primitive-action prerequisites. Sharing has no distance requirement in this environment.
- Future collection-tool dependencies for `Navigator`. Missing a tool does not make movement illegal. Choosing the same material as another agent is not itself a conflict.

Checks return `satisfied`, `violated`, or `unknown`. Missing observations stay unknown. A teammate's planned facility is not treated as already built. The checker reads the provided observations near each agent and does not query additional simulator state. It does not predict complete future trajectories or implement CUBE constraints.

In `repair` mode, concrete violations are sent to affected AI agents and inventory holders who could help. They replan concurrently from the same observation snapshot, for at most two extra rounds. Rules do not rewrite plans directly. Existing valid navigation is not repeatedly replanned on every movement step just because a future tool is missing.

After the limit, operations with verified unmet immediate prerequisites are held as `noop`, and the agent replans on the next step. Future navigation dependencies and unknown checks do not hold movement. Held candidates are not added to execution memory/history; only the final accepted planning response is consolidated. The shared thinking processor also supplies these checks to the existing GUI callers.

A failed repair reply keeps the last valid candidate; the error is logged and included in the next repair round if budget remains. It does not add extra repair rounds or interrupt other agents.

```powershell
python scripts\smoke_coordination.py
python run.py --agent_num 2 --step_num 10 --coordination_mode repair --repair_rounds 2 --results_dir results\repair_smoke --no_plot
```

For a diagnosis-only comparison, use `--coordination_mode diagnose`; this adds no repair calls or action holds. `--coordination_mode off` keeps the original planning behavior. These switches are exposed by the headless entry point.

Each headless run saves `coordination.jsonl` under its round's output directory, with plans, per-rule checks, replanned agents, and held agents. Its `step` follows the runner's existing zero-based observation index. Repair rounds count planning revisions; provider/API retry settings are separate. Terminal headless steps do not request new plans or repairs.

## Relation to the original DAMCS project

This repository is a reproduction and engineering adaptation, not the official DAMCS release. It follows the same high-level idea from the paper and project page: decentralized LLM agents cooperate in Multi-Agent Crafter using structured memory and communication.

Main differences from the paper/original upstream project:

- The code is organized as a runnable local reproduction with additional setup, smoke-test, and provider-configuration scripts.
- Agents are implemented on the AgentScope framework instead of calling the OpenAI SDK directly (see [AgentScope integration](#agentscope-integration)).
- The README documents a Windows/Conda workflow and Qwen/DashScope usage in addition to Azure OpenAI/OpenAI style configuration.
- Local API keys are expected to live in `.env`; secret files and generated outputs are ignored by Git.
- The default command is a practical DAMCS-style six-agent run, while exact paper metrics may require matching the paper's full experimental settings, model choices, random seeds, and evaluation protocol.
- Some GUI and helper scripts are retained for visualization and debugging, so this repository is closer to a reproducible course/project implementation than a minimal paper artifact.

## Verification without an LLM

Check the AgentScope integration without credentials or LLM API requests. The check uses local mock HTTP responses with the real AgentScope model adapters, message formatter, and agents:

```powershell
python scripts\smoke_agentscope.py
```

It checks OpenAI/Azure/Qwen chat and embeddings, multimodal messages, reply hooks, concurrent planning, skipped agents, failure isolation, JSON correction, and the async bridge. It does not check provider connectivity or model availability; use `smoke_llm.py --mode both --strict` for those checks.

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
| `ModuleNotFoundError: No module named 'agentscope'` | Run `conda run -n mcrafter python -m pip install -r requirements.txt` and launch the project with the `mcrafter` interpreter. |
| `ImportError: cannot import name 'streamablehttp_client'` when importing `agentscope` | Install the pinned MCP 1.x dependency with `python -m pip install -r requirements.txt`. |
| `AttributeError: 'AsyncCompletions' object has no attribute 'parse'` | Update the OpenAI SDK using `python -m pip install -r requirements.txt`, then run `python scripts\smoke_agentscope.py`. |
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

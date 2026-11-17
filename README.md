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
|-- run_cube.py                # COOP2 CUBE runner using the same review controller
|-- third_party/coop2_cube/     # Pinned upstream CUBE rules and MIT license
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

### Other vendors and named model profiles

`LLM_PROVIDER=compatible` targets any OpenAI-compatible chat endpoint (DeepSeek, Kimi, GLM, OpenAI) with `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_CHAT_MODEL`. Such endpoints return JSON that is validated locally and retried with the error (`LLM_JSON_MODE=json_object`). `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` and `EMBEDDING_MODEL` send embedding calls to a separate endpoint, so chat models can change while the embedding model stays fixed. `LLM_GENERATE_KWARGS` adds JSON request fields such as temperature to every chat call.

For experiments, use the named profiles in [config/model_profiles.json](config/model_profiles.json) instead of setting variables by hand. The file holds model names, endpoints and prices; keys stay in environment variables, `local_config/api_keys.json`, or the existing Qwen CSV.

```powershell
python scripts\run_with_model.py --profile qwen        # print the resolved configuration, no keys
python scripts\run_with_model.py --profile kimi run.py --condition A --step_num 20 --results_dir results\trial_kimi --no_plot
python scripts\run_experiments.py --model_profiles qwen deepseek --conditions A B C --seeds 4 --dry_run
```

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
| `--scope_mode` | `suggest` | In `repair` mode, generate and log verified repair-scope suggestions; `off` skips scope search. |
| `--scope_max_agents` | none | Cap K on AI agents a repair may change. Default: no cap; the search deepens by agent count and keeps the scope that covers the most targets with the fewest agents. Set `1` or `2` for ablations. Separate from repair rounds. |
| `--review_max_agents` | `2` | Most agents one human review may open. A smallest sufficient repair above this skips review and uses the automatic fallback. |
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

Use `--condition A` for planning with common execution protection, or `--condition B` to add at most two targeted replanning rounds. The headless runner applies the same immediate-prerequisite guard in every condition, including `diagnose` and `off`. The lower-level coordination API retains its opt-in guard parameter for existing GUI callers.

Each headless run saves `coordination.jsonl` under its round's output directory, with plans, per-rule checks, replanned agents, and held agents. Its `step` follows the runner's existing zero-based observation index. Repair rounds count planning revisions; provider/API retry settings are separate. Terminal headless steps do not request new plans or repairs.

### Repairable scope suggestions

After targeted replanning, `memory_system/repair_scopes.py` enumerates finite edits to the current plans and recommends a joint repair scope. The first version covers crafting-material deficits and missing collection tools when the declared target material is already in front of the agent. It checks one environment step, including transfers before primitive actions.

The edited agents can differ from the agents with reported problems. A provider sharing wood can restore another agent's unchanged crafting plan. The scope records the actual changed fields separately from the agents whose facts and commitments are needed as evidence.

Each candidate must produce its fixed manufacturing or collection event and preserve verified work outside the edited scope. An all-`noop` plan does not repair a manufacturing goal. Shared stock is consumed in execution order, so independently feasible repairs cannot claim the same unit twice. Checks return `repairable`, `infeasible` within the declared finite domain, or `unknown` when relevant evidence or effects are missing. The allowed domain and horizon accompany the result.

The projection models the submitted-state player action pass, using recipes and the agents' submitted observations. Existing navigation contributes only its next primitive move; future trajectories, survival and later autonomous-object changes are outside this model's guarantee.

`--scope_mode suggest` adds no model calls and records candidates without applying them. `--scope_mode auto` verifies and commits the selected joint patch. `--scope_mode human` offers a budgeted plan review before the same automatic fallback. Diagnosis-only and coordination-off modes skip scope search.

`coordination.jsonl` includes `scope_analysis`: targets, finite domains, per-scope results, minimal candidates, a selected witness (or a reason for no recommendation), assignment count and search time. Selection prioritizes covered immediate targets, then direct collection targets, then fewer edited agents and fewer fields. It can cover part of a snapshot's targets; other unresolved operations retain their existing execution policy.

```powershell
python scripts\smoke_repair_scopes.py
python scripts\evaluate_scope_examples.py --results_dir results\scope_examples
python run.py --agent_num 2 --step_num 10 --scope_mode suggest --results_dir results\scope_smoke --no_plot
```

The example comparison uses controlled cases and verifies selected witnesses with real Crafter execution. It compares problem-owner ordering, a complete-related-group coverage reference, and verified scope selection. It measures structural repairability and search cost; human benefit and full-task improvements need separate experiments.

## Budgeted plan review and experiments

The controller in `memory_system/intervention.py` keeps edit permissions S, review evidence E, and waiting agents P separate. It checks the whole submitted patch before applying any edit. The executor pauses both primitive actions and the transfers that happen before `env.step`.

| Condition | Behavior |
| --- | --- |
| A | Autonomous planning and the common execution guard |
| B | A plus at most two targeted replanning rounds |
| C | B plus verified scope search and automatic execution |
| H1 | B plus problem-owner review, with C's automatic fallback |
| H2 | B plus verified repair-scope review, with the same fallback |

Renamed on 2026-11-17: the old related-group condition (then called H1, all fields open) was removed because its free-form group edits were hard to quantify, and the problem-owner condition was renamed from H0 to H1. All conditions use submitted observations and the same environment rules. The candidate domain is shared; the selected edit scope controls which agents and fields the reviewer can change.

```powershell
# Automatic scope execution
python run.py --condition C --agent_num 6 --step_num 350 --seed 4 --results_dir results\auto --no_plot

# Open the printed localhost URL to approve, preview, edit or skip a request
python run.py --condition H2 --review_max_agents 2 --pause_mode local --results_dir results\human --no_plot

# Preview a paired-seed matrix without calling models
python scripts\run_experiments.py --conditions A B C H1 H2 --seeds 4 5 6 --dry_run

# A scripted reviewer checks the pipeline; its replies are labelled as scripted
python scripts\run_experiments.py --conditions C H2 --seeds 4 --step_num 10 --reviewer scripted --results_dir results\pipeline

# Offline comparison on identical captured failure snapshots
python scripts\evaluate_frozen_scopes.py --input results\pipeline
```

Human review has no per-episode request budget by default. It asks only at critical moments, judged by three environment-independent signals (`--review_trigger signals`): verified repairs would divert different agents, so someone must choose (`choice`); no repair can be verified because facts are missing (`unknown`); or team progress has not grown for a tenth of the episode (`--stall_fraction 0.1`: 35 of 350 Crafter steps, 3 of 30 CUBE steps; `--stall_steps` sets a fixed number instead; `stalled`). The same problem is not asked again within `--review_cooldown` steps (default 10), and everything else uses the automatic fallback. `--review_trigger stage` uses a fixed per-environment list instead (Crafter: iron pickaxe, iron or diamond blocked; CUBE: any failed push); `--review_trigger all` asks about every problem (for practice); `--human_requests` sets an optional cap. `--scope_max_agents` is the search cap K (none by default); `--review_max_agents` caps what one review may open. `--review_timeout` is wall time; `--review_step_seconds` paces environment ticks while waiting. Global and local waiting both advance the environment, with held agents executing `noop`. `--pause_mode none` continues execution and rechecks replies. `--revalidation full` always checks the whole team; `incremental` checks dependencies and reuses an unchanged sufficient certificate when possible.

Crafter compact evidence removes inventory/view groups only when the declared one-step verdict and existing protection obligations remain provable. Minimality is by deletion of these groups relative to fixed plans, positions and permissions; it is not a global minimum or a guarantee of human understanding. CUBE retains its complete observed board as one evidence group.

Each episode saves `config.json` (seed, condition, source hashes and model names), `model_usage.json`, `events.jsonl`, `intervention_summary.json`, immutable requests under `reviews/`, and replayable submitted-state snapshots under `failures/`. Logs distinguish approval, commit, actual target recovery and final task success. Batch runs export `episodes.json` and `episodes.csv`, retaining failed episodes and process exit codes. Use `--summarize_only` to regenerate the exports. Crafter episodes save their whole running state to `checkpoint.pkl` every `--checkpoint_every` steps (default 5); rerunning the same command with `--resume` skips finished episodes, continues interrupted Crafter episodes from their checkpoint, and reruns the rest, keeping failed attempts as `<dir>.failedN`. A checkpoint written by different code is refused.

The planned result tables, their commands and the exported fields are in [docs/experiment_tables_2026-11-17.md](docs/experiment_tables_2026-11-17.md). `--suite ablation` runs the component ablations (no replanning, any-feasible scope selection, and the scripted-review variants for waiting and rechecking, including `--revalidation none`). Other batch sweeps: `--suite capacity --conditions C H2`, `--suite evidence --conditions H2`, and `--suite waiting --conditions H2`. Frozen-snapshot evaluation also compares individual problem witnesses with joint consistency; it is an offline diagnostic ablation, not a full-task performance claim. Use an episode as the statistical unit and record reviewer identities separately for a real human study.

### CUBE transfer check

`third_party/coop2_cube` contains the actual [COOP2 environment source](https://github.com/HappyEureka/coop2-llm-mas), pinned in `UPSTREAM.json`, with its MIT license. The package entry point is trimmed; the environment and task rules are unchanged. The adapter reconstructs its transition model from public agent/block observations and checks simultaneous force, collision and delivery using those rules.

```powershell
python run_cube.py --condition C --agent_num 3 --grid_size 9 --block_specs 2:1 --step_num 30 --results_dir results\cube --no_plot
python scripts\run_experiments.py --environment cube --conditions A B C H2 --agent_num 3 --grid_size 9 --block_specs 2:1 --seeds 4 5 6 --step_num 30 --dry_run
```

The CUBE planner is an AgentScope agent with a small action schema. It uses the common bounded review/commit protocol and CUBE-specific push diagnostics; it does not import Crafter's crafting goals. Neither environment adapter promises general multi-step planning or future unobserved outcomes.

Implementation and validation details: [completed framework](docs/implementation_2026-11-16.md).

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

Run the complete mechanism suite without provider calls:

```powershell
python -B -m unittest discover -s scripts -p "smoke_*.py"
```

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

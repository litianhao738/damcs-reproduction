from tqdm import tqdm
from utils import (
    AgentActionProcessor, 
    EnvironmentManager, 
    SimulationLogger,
    AgentStateManager,
    SimulationContextManager,
    AgentThinkingProcessor,
    get_arg_parser
)
from memory_system.agent import ActionStatus
import crafter
from memory_system.agent import Agent, HumanAgent
from plot import Plotter
import json
import pandas as pd
import os
from pathlib import Path
import time
from memory_system.intervention import InterventionController, write_json
from memory_system.review_server import ReviewServer
from memory_system.team_board import TeamBoard
from memory_system.llm_api import MODEL_USAGE, get_model_usage
from memory_system.checkpoint import (
    CheckpointError, load_checkpoint, remove_checkpoint, save_checkpoint, trim_after_checkpoint,
)
from memory_system.experiments import apply_condition, experiment_config, ScriptedReviewer

MAX_STEPS = 350
N_PLAYERS = 6

class CompleteMulitAgentSimulation:
    """Complete simulation that handles the full agent processing pipeline"""
    
    def __init__(self, agents, env, max_steps, n_players,
                 coordination_mode="repair", repair_rounds=2, coordination_log=None,
                 scope_mode="suggest", scope_max_agents=None, intervention=None,
                 execution_guard=None, review_step_seconds=1.0, reviewer=None, scope_selection="minimal",
                 checkpoint_dir=None, checkpoint_every=0, source_sha256=None, resume_state=None,
                 team_board=False):
        """``checkpoint_every``: save the whole running state to ``checkpoint_dir`` every
        that many steps (0 disables). ``resume_state``: a loaded checkpoint to continue from;
        its environment, agents and controller must be the ones passed in. ``team_board``:
        show every agent the shared team board (``memory_system.team_board``)."""
        self.checkpoint_dir, self.checkpoint_every = checkpoint_dir, checkpoint_every
        self.source_sha256, self.resume_state = source_sha256, resume_state
        self.agents = agents
        self.env = env
        self.max_steps = max_steps
        self.n_players = n_players
        self.coordination_mode = coordination_mode
        self.repair_rounds = repair_rounds
        self.coordination_log = coordination_log
        self.scope_mode = scope_mode
        self.scope_max_agents = scope_max_agents
        self.scope_selection = scope_selection
        self.intervention = intervention
        if self.intervention is None and scope_mode in ('auto', 'human'):
            self.intervention = InterventionController(agents[0].agent_dir.parent,
                                                       mode=scope_mode, search_agents=scope_max_agents,
                                                       selection=scope_selection, step_limit=max_steps)
        self.execution_guard = execution_guard
        self.review_step_seconds = review_step_seconds
        self.reviewer = reviewer
        
        # Initialize processors
        self.action_processor = AgentActionProcessor()
        self.env_manager = EnvironmentManager()
        self.reporter = SimulationLogger()
        self.agent_state_manager = AgentStateManager()
        self.simulation_context_manager = SimulationContextManager(self.n_players)
        if team_board:
            self.simulation_context_manager.board = TeamBoard(self.intervention)
        self.agent_thinking_processor = AgentThinkingProcessor()

    def run_simulation(self):
        """Run the complete simulation with progress tracking"""
        if self.resume_state is not None:
            loop = self.resume_state["loop"]
            total_step, _step, done = loop["total_step"], loop["step"], loop["done"]
            obs, info, rewards = loop["obs"], loop["info"], loop["rewards"]
            episode_number = loop["episode_number"]
            usage_before = self.resume_state["usage_before"]
            # Model usage continues from the checkpoint; calls made after it by the
            # interrupted run are not counted.
            MODEL_USAGE.clear()
            MODEL_USAGE.update(self.resume_state["model_usage"])
        else:
            total_step = 0
            _step = 0
            obs = self.env.reset()
            done = False
            episode_number = 0
            info = None
            usage_before = get_model_usage()

        with tqdm(total=self.max_steps, desc="Progress", initial=total_step) as bar:
            while total_step < self.max_steps and not done:
                if self.intervention is not None and info is not None:
                    if self.reviewer is not None:
                        self.reviewer.poll(self.intervention, _step)
                    resumed = self.intervention.before_actions(self.agents, dict(enumerate(info)))
                    for reply in resumed:
                        agent = self.agents[reply['id']]
                        if not reply.get('deferred', False) and agent.pending_plan_response is not None:
                            agent.create_experience()
                    self.agent_thinking_processor.update_agents_from_responses(self.agents, resumed)
                    if self.intervention.pending is not None:
                        time.sleep(self.review_step_seconds)
                self.reporter.print_step_header(total_step, _step)
                
                # Process each agent's actions with specific logic
                self.action_processor.process_all_agent_actions(
                    self.agents, self.env, self.n_players,
                    event_sink=(lambda kind, **data: self.intervention.event(kind, _step, **data))
                    if self.intervention is not None else None,
                )
                
                # Collect actions and step environment
                agents_actions = self.action_processor.collect_agent_actions(
                    self.agents, self.n_players
                )
                
                before_info = info
                if self.intervention is not None:
                    self.intervention.event('action_attempt', _step, actions=agents_actions)
                obs, rewards, done, info = self.env_manager.step_environment(
                    self.env, agents_actions
                )
                if self.intervention is not None:
                    self.intervention.observe(before_info, info, self.agents, _step)
                self.env_manager.update_crafting_stations(self.agents, self.env)
                tool_avaliablity_info = self.env_manager.get_tool_availability_info(self.env, self.n_players)

                self.agent_state_manager.update_all_agent_states(
                    self.agents, obs, _step, self.env, info, episode_number
                )
                if not done and self.intervention is not None:
                    # Locked progress tasks set their agents' plans before anyone plans.
                    self.intervention.prepare(self.agents, dict(enumerate(info)))
                if not done:
                    agents_with_new_thought = self.agent_state_manager.identify_agents_needing_thought(self.agents, info)
                    agents_contexts = self.simulation_context_manager.create_agent_contexts(self.agents, info, _step)
                    agents_responses = self.agent_thinking_processor.process_agent_thinking_parallel(
                        self.agents, agents_contexts, info,
                        coordination_mode=self.coordination_mode,
                        max_repair_rounds=self.repair_rounds,
                        coordination_log=self.coordination_log,
                        scope_mode=self.scope_mode,
                        scope_max_agents=self.scope_max_agents,
                        intervention=self.intervention,
                        execution_guard=self.execution_guard,
                        scope_selection=self.scope_selection,
                    )
                    self.agent_thinking_processor.update_agents_from_responses(
                        self.agents, agents_responses
                    )
                    self.reporter.show_step_report(self.agents, agents_with_new_thought)
                # Update counters and progress
                total_step += 1
                _step += 1
                bar.update(1)
                if (self.checkpoint_dir is not None and self.checkpoint_every
                        and total_step % self.checkpoint_every == 0 and not done
                        and total_step < self.max_steps
                        and (self.intervention is None or self.intervention.pending is None)):
                    save_checkpoint(self.checkpoint_dir, {
                        "source_sha256": self.source_sha256, "step": total_step,
                        "loop": {"total_step": total_step, "step": _step, "done": done, "obs": obs,
                                 "info": info, "rewards": rewards, "episode_number": episode_number},
                        "usage_before": usage_before, "model_usage": dict(MODEL_USAGE),
                        "env": self.env, "agents": self.agents, "intervention": self.intervention,
                    })
        
        self.reporter.print_simulation_complete()
        # self.reporter.print_stats(self.agents)
        df = self.reporter.log_stats(self.agents)
        success = any(row['achievements'].get('collect_diamond', 0) > 0 for row in info)
        reason = 'success' if success else 'all_dead' if all(row['dead'] for row in info) else 'step_limit'
        intervention_summary = self.intervention.finish(info, _step, reason) if self.intervention is not None else None
        usage_after = get_model_usage()
        usage = {key: value - usage_before.get(key, 0) for key, value in usage_after.items()}
        if self.intervention is not None:
            write_json(self.intervention.output_dir / 'model_usage.json', usage)
        if self.checkpoint_dir is not None:
            # The episode finished; its checkpoint is no longer needed.
            remove_checkpoint(self.checkpoint_dir)


        return {
            'total_steps': total_step,
            'completed': done,
            'final_obs': obs,
            'final_rewards': rewards,
            'final_info': info,
            'stats_df': df,
            'termination_reason': reason, 'intervention_summary': intervention_summary,
            'model_usage': usage,
        }
    
    def get_agent_summary(self):
        """Get a summary of current agent states"""
        return [
            {
                'id': agent.id,
                'op': agent.op,
                'action': agent.action,
                'action_status': agent.action_status,
                'rss_to_collect': getattr(agent, 'rss_to_collect', 'N/A'),
                'rss_to_share': getattr(agent, 'rss_to_share', 'N/A'),
                'target_agent_id': getattr(agent, 'target_agent_id', 'N/A')
            }
            for agent in self.agents
        ]

def initialize_agents(human_agent_ids=None, n_players=None, results_dir="results"):
    if human_agent_ids is None:
        human_agent_ids = []
    elif isinstance(human_agent_ids, str) and human_agent_ids.strip():
        # Parse comma-separated string of agent IDs
        human_agent_ids = [int(x.strip()) for x in human_agent_ids.split(',') if x.strip().isdigit()]
    elif not isinstance(human_agent_ids, list):
        human_agent_ids = []
    
    agents = []
    for i in range(n_players):
        if i in human_agent_ids:
            agents.append(HumanAgent(id=i, kg_update_freq=10, results_dir=results_dir))
        else:
            agents.append(Agent(id=i, kg_update_freq=10, results_dir=results_dir))
    return agents

def initialize_environment(
        num_steps=None, n_players=None, global_map_dir=None,
        global_map_start=0, global_map_unit=8, seed=4):
    return crafter.Env(
        length=num_steps,
        n_players=n_players,
        seed=seed,
        global_map_dir=global_map_dir,
        global_map_start=global_map_start,
        global_map_unit=global_map_unit,
    )


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    apply_condition(args)
    num_rounds = args.num_of_turn if hasattr(args, 'num_of_turn') else 1
    num_steps = args.step_num if hasattr(args, 'step_num') else 350
    human_agent_ids = args.human_agents if hasattr(args, 'human_agents') else None
    agent_num = args.agent_num if hasattr(args, 'agent_num') else 1
    results_dir = args.results_dir if hasattr(args, 'results_dir') else 'results'
    os.makedirs(results_dir, exist_ok=True)
    global_map_dir = (
        os.path.join(results_dir, 'global_maps')
        if getattr(args, 'save_global_maps', False) else None
    )
    all_stats = []
    for round_idx in range(num_rounds):
        print(f"=== Simulation Round {round_idx+1} ===")
        round_results_dir = (
            os.path.join(results_dir, f'round_{round_idx + 1}')
            if num_rounds > 1 else results_dir
        )
        round_dir = Path(round_results_dir)
        config = {**experiment_config(args, 'crafter'), 'repeat_id': args.repeat_id + round_idx}
        save_path = os.path.join(results_dir, f"stats_round_{round_idx+1}_agent_{agent_num}.csv")
        if args.resume and os.path.exists(save_path) and (round_dir / 'intervention_summary.json').exists():
            print(f"Round {round_idx+1} already finished; reusing its stats.")
            all_stats.append(pd.read_csv(save_path))
            continue
        state = None
        if args.resume:
            try:
                state = load_checkpoint(round_dir, config['source_sha256'])
            except CheckpointError as error:
                raise SystemExit(str(error))
        if state is not None:
            env, agents, intervention = state['env'], state['agents'], state['intervention']
            # Forget what the interrupted run logged after the checkpoint; those steps run again.
            trim_after_checkpoint(round_dir, state['loop']['step'] - 1, intervention.issued)
            config_path = round_dir / 'config.json'
            previous = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else config
            previous.setdefault('resumes', []).append({'from_step': state['step'], 'time': time.time()})
            write_json(round_dir / 'config.json', previous)
            print(f"Resuming round {round_idx+1} from the checkpoint at step {state['step']}.")
        else:
            if args.resume:
                # No checkpoint: discard partial logs of an interrupted attempt and start over.
                trim_after_checkpoint(round_dir, -1, 0)
            agents = initialize_agents(
                human_agent_ids=human_agent_ids, n_players=agent_num,
                results_dir=round_results_dir,
            )
            for agent in agents:
                # The "mechanisms only" control: idle plans, no model calls.
                agent.llm_enabled = args.agent_llm == 'on'
            round_global_map_dir = global_map_dir
            if global_map_dir and num_rounds > 1:
                round_global_map_dir = os.path.join(
                    global_map_dir, f'round_{round_idx + 1}')
            env = initialize_environment(
                num_steps=num_steps,
                n_players=agent_num,
                global_map_dir=round_global_map_dir,
                global_map_start=args.global_map_start,
                global_map_unit=args.global_map_unit,
                seed=args.seed,
            )
            intervention = InterventionController(
                round_results_dir, mode=args.scope_mode, strategy=args.review_strategy,
                capacity=args.review_max_agents, search_agents=args.scope_max_agents,
                selection=args.scope_selection, request_budget=args.human_requests,
                review_trigger=args.review_trigger, review_cooldown=args.review_cooldown, stall_steps=args.stall_steps,
                stall_fraction=args.stall_fraction, step_limit=num_steps,
                timeout=args.review_timeout, pause_mode=args.pause_mode, evidence_mode=args.evidence_mode,
                revalidation=args.revalidation, progress_mode=args.progress_mode,
                progress_commit=args.progress_commit == 'on', slow_margin=args.slow_margin,
                progress_pause=args.progress_pause == 'on', progress_prestage=args.progress_prestage == 'on',
            )
            write_json(round_dir / 'config.json', config)
        review_server = ReviewServer(intervention.review_dir, args.review_port) \
            if args.scope_mode == 'human' and args.reviewer == 'human' and args.review_port else None
        if review_server is not None:
            print('Plan review: ' + review_server.start())
        simulation = CompleteMulitAgentSimulation(
            agents, env, num_steps, agent_num,
            coordination_mode=args.coordination_mode,
            repair_rounds=args.repair_rounds,
            coordination_log=os.path.join(round_results_dir, "coordination.jsonl"),
            scope_mode=args.scope_mode,
            scope_max_agents=args.scope_max_agents, scope_selection=args.scope_selection,
            intervention=intervention, execution_guard=True,
            review_step_seconds=args.review_step_seconds,
            reviewer=ScriptedReviewer(args.scripted_decision, args.scripted_delay_steps)
                if args.reviewer == 'scripted' else None,
            checkpoint_dir=round_results_dir, checkpoint_every=args.checkpoint_every,
            source_sha256=config['source_sha256'], resume_state=state,
            team_board=args.team_board == 'on',
        )
        try:
            results_df = simulation.run_simulation()['stats_df']
        finally:
            if review_server is not None:
                review_server.close()
        # Save stats, file name contains round number and agent number
        results_df.to_csv(save_path, index=False)
        all_stats.append(results_df)
    # combine all rounds results
    final_df = pd.concat(all_stats, ignore_index=True)
    all_rounds_path = os.path.join(results_dir, "all_rounds_stats.csv")

    # if file exists, append to it, otherwise create a new file
    final_df.to_csv(
        all_rounds_path,
        mode='a',
        header=not os.path.exists(all_rounds_path),
        index=False
    )
    print("Appended results to all_rounds_stats.csv")
    if not getattr(args, 'no_plot', False):
        plotter = Plotter()
        plotter.plot([all_rounds_path], legend_labels=['Six Agents with Communication'])


if __name__ == "__main__":
    main()

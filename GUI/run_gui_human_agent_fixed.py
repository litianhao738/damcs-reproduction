#!/usr/bin/env python3
# 模块说明：提供经过修正的人工智能体图形界面实现。
"""
Fixed GUI for HumanAgent integration - resolves black screen issue
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
try:
    import pygame
except ImportError:
    print('Please install the pygame package to use the GUI.')
    raise
from PIL import Image

import crafter
from memory_system.agent import HumanAgent, Agent
from run import initialize_agents
from utils import AgentActionProcessor, EnvironmentManager, SimulationLogger, AgentStateManager, SimulationContextManager, AgentThinkingProcessor, ActionStatus


class FixedHumanAgentGUI:
    """Fixed GUI class for HumanAgent integration - no black screen"""
    
    def __init__(self, human_agent_ids=None, n_players=3, max_steps=350):
        # 变量说明（human_agent_ids）：由用户手动控制的智能体编号集合。
        # 变量说明（n_players）：共享环境中的玩家总数。
        # 变量说明（max_steps）：允许本轮仿真执行的最大步数。
        # 变量说明（self.human_agent_ids）：由用户手动控制的智能体编号集合。
        self.human_agent_ids = human_agent_ids or []
        # 变量说明（self.n_players）：共享环境中的玩家总数。
        self.n_players = n_players
        # 变量说明（self.max_steps）：允许本轮仿真执行的最大步数。
        self.max_steps = max_steps
        
        # Initialize agents
        # 变量说明（self.agents）：参与当前仿真的全部智能体对象列表。
        self.agents = initialize_agents(human_agent_ids=human_agent_ids, n_players=n_players)
        
        # Initialize environment
        # 变量说明（self.env）：当前 Crafter 仿真环境实例。
        self.env = crafter.Env(length=max_steps, n_players=n_players, seed=4)
        self.env.reset()
        
        # Initialize processors
        # 变量说明（self.action_processor）：把 LLM 高层决策转换成环境动作的处理器。
        self.action_processor = AgentActionProcessor()
        # 变量说明（self.env_manager）：统一推进环境并同步工作台、熔炉等共享状态的管理器。
        self.env_manager = EnvironmentManager()
        # 变量说明（self.reporter）：负责终端步骤报告和实验统计输出的日志器。
        self.reporter = SimulationLogger()
        # 变量说明（self.agent_state_manager）：把环境观察和状态同步到各智能体的管理器。
        self.agent_state_manager = AgentStateManager()
        # 变量说明（self.simulation_context_manager）：为每个智能体组装个性化协作提示的管理器。
        self.simulation_context_manager = SimulationContextManager(n_players)
        # 变量说明（self.agent_thinking_processor）：并行调用各 AI 智能体思考流程的处理器。
        self.agent_thinking_processor = AgentThinkingProcessor()
        
        # GUI state
        # 变量说明（self.current_step）：`__init__` 中当前处理的时间步。
        self.current_step = 0
        # 变量说明（self.running）：控制 GUI 主循环是否继续运行的布尔标记。
        self.running = True
        # 变量说明（self.paused）：仿真是否暂停推进的布尔标记。
        self.paused = False
        # 变量说明（self.current_human_agent）：`__init__` 中当前处理的智能体。
        self.current_human_agent = None
        # 变量说明（self.waiting_for_human_input）：GUI 是否正在等待人工智能体输入动作参数。
        self.waiting_for_human_input = False
        # 变量说明（self.input_mode）：人工输入流程当前正在收集的字段阶段。
        self.input_mode = False
        # 变量说明（self.input_buffer）：尚未提交给人工智能体的键盘输入字符缓冲区。
        self.input_buffer = ""
        # 变量说明（self.input_prompt）：GUI 提示用户输入当前字段的说明文本。
        self.input_prompt = ""
        # 变量说明（self.input_field）：GUI 当前已经输入、尚未提交的字符串。
        self.input_field = ""
        # 变量说明（self.input_type）：人工输入字段需要转换成的目标数据类型。
        self.input_type = ""  # 'op', 'collect', 'share', 'target'
        
        # Pygame setup
        # 变量说明（self.window_size）：GUI 主窗口的宽高像素值。
        self.window_size = (800, 600)
        # 变量说明（self.fps）：界面刷新或视频编码的每秒帧数。
        self.fps = 5
        pygame.init()
        # 变量说明（self.screen）：Pygame 创建的主显示窗口表面。
        self.screen = pygame.display.set_mode(self.window_size)
        # 变量说明（self.clock）：控制 GUI 刷新频率的 Pygame 时钟。
        self.clock = pygame.time.Clock()
        
        # Key mappings
        # 变量说明（self.keymap）：键盘按键到 Crafter 动作编号的映射。
        self.keymap = {
            pygame.K_a: 'move_left',
            pygame.K_d: 'move_right',
            pygame.K_w: 'move_up',
            pygame.K_s: 'move_down',
            pygame.K_SPACE: 'do',
            pygame.K_TAB: 'sleep',
            pygame.K_r: 'place_stone',
            pygame.K_t: 'place_table',
            pygame.K_f: 'place_furnace',
            pygame.K_p: 'place_plant',
            pygame.K_1: 'make_wood_pickaxe',
            pygame.K_2: 'make_stone_pickaxe',
            pygame.K_3: 'make_iron_pickaxe',
            pygame.K_4: 'make_wood_sword',
            pygame.K_5: 'make_stone_sword',
            pygame.K_6: 'make_iron_sword',
            pygame.K_LSHIFT: 'switch_player',
            pygame.K_p: 'pause',
            pygame.K_h: 'human_action',
        }
        
        print('Fixed GUI Actions:')
        # 变量说明（action）：当前智能体准备执行的环境动作。
        # 变量说明（key）：当前字典条目的键。
        for key, action in self.keymap.items():
            print(f'  {pygame.key.name(key)}: {action}')
        print('  h: Human agent action selection')
        print('  p: Pause/Resume simulation')
        print('  ESC: Quit')
        print('  ENTER: Confirm input (when in input mode)')
    
    def render(self):
        """Render the game state"""
        # Render environment
        # 变量说明（image）：当前读取、处理或渲染的图像数组。
        image = self.env.render(self.window_size)
        # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render UI overlay
        self.render_ui_overlay()
        
        # Render input interface if in input mode
        if self.input_mode:
            self.render_input_interface()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def render_ui_overlay(self):
        """Render UI overlay with agent information"""
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 24)
        
        # Step counter
        # 变量说明（step_text）：`render_ui_overlay` 中当前处理的时间步、文本。
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
        self.screen.blit(step_text, (10, 10))
        
        # Agent status
        # 变量说明（y_offset）：当前元素相对布局起点的纵向像素偏移量。
        y_offset = 40
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        for i, agent in enumerate(self.agents):
            # 变量说明（agent_type）：`render_ui_overlay` 中当前处理的智能体。
            agent_type = "Human" if hasattr(agent, 'is_human') and agent.is_human else "AI"
            # 变量说明（status）：调用方准备写入对象的最新执行状态。
            status = "Waiting" if agent.action_status == ActionStatus.DONE else "Working"
            # 变量说明（color）：`render_ui_overlay` 中当前处理的颜色。
            color = (255, 255, 0) if hasattr(agent, 'is_human') and agent.is_human else (255, 255, 255)
            
            # 变量说明（agent_text）：`render_ui_overlay` 中当前处理的智能体、文本。
            agent_text = font.render(f'Agent {i} ({agent_type}): {status}', True, color)
            self.screen.blit(agent_text, (10, y_offset))
            
            if hasattr(agent, 'op') and agent.op:
                # 变量说明（op_text）：`render_ui_overlay` 中当前处理的文本。
                op_text = font.render(f'  Op: {agent.op}', True, (200, 200, 200))
                self.screen.blit(op_text, (20, y_offset + 20))
            
            y_offset += 50
        
        # Pause indicator
        if self.paused:
            # 变量说明（pause_text）：`render_ui_overlay` 中当前处理的文本。
            pause_text = font.render('PAUSED - Press P to resume', True, (255, 0, 0))
            self.screen.blit(pause_text, (10, self.window_size[1] - 30))
        
        # Input mode indicator
        if self.input_mode:
            # 变量说明（input_text）：`render_ui_overlay` 中当前处理的文本。
            input_text = font.render('INPUT MODE - Type your response', True, (0, 255, 0))
            self.screen.blit(input_text, (10, self.window_size[1] - 60))
    
    def render_input_interface(self):
        """Render input interface overlay"""
        # Semi-transparent overlay
        # 变量说明（overlay）：叠加到游戏画面上、用于增强可读性的半透明图层。
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 24)
        
        # Input prompt
        # 变量说明（prompt_text）：`render_input_interface` 中当前处理的文本。
        prompt_text = font.render(self.input_prompt, True, (255, 255, 255))
        self.screen.blit(prompt_text, (50, 200))
        
        # Input field
        # 变量说明（input_text）：`render_input_interface` 中当前处理的文本。
        input_text = font.render(f'Input: {self.input_field}', True, (255, 255, 0))
        self.screen.blit(input_text, (50, 230))
        
        # Instructions
        # 变量说明（instruction_text）：`render_input_interface` 中当前处理的文本。
        instruction_text = font.render('Press ENTER to confirm, ESC to cancel', True, (200, 200, 200))
        self.screen.blit(instruction_text, (50, 260))
    
    def handle_events(self):
        """Handle pygame events"""
        # 变量说明（event）：`handle_events` 中当前处理的事件。
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.input_mode:
                    self.handle_input_event(event)
                else:
                    self.handle_normal_event(event)
    
    def handle_normal_event(self, event):
        # 变量说明（event）：`handle_normal_event` 中当前处理的事件。
        """Handle events when not in input mode"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_h:
            self.start_human_action_selection()
        elif event.key in self.keymap:
            # 变量说明（action）：当前智能体准备执行的环境动作。
            action = self.keymap[event.key]
            self.handle_action(action)
    
    def handle_input_event(self, event):
        # 变量说明（event）：`handle_input_event` 中当前处理的事件。
        """Handle events when in input mode"""
        if event.key == pygame.K_ESCAPE:
            # Cancel input
            self.input_mode = False
            self.input_field = ""
            self.input_type = ""
        elif event.key == pygame.K_RETURN:
            # Confirm input
            self.process_input()
        elif event.key == pygame.K_BACKSPACE:
            # Delete character
            self.input_field = self.input_field[:-1]
        else:
            # Add character
            if event.unicode.isprintable():
                self.input_field += event.unicode
    
    def handle_action(self, action):
        # 变量说明（action）：当前智能体准备执行的环境动作。
        """Handle game actions"""
        if action == 'switch_player':
            # Switch between human agents
            # 变量说明（agent）：当前正在处理的单个智能体对象。
            # 变量说明（human_agents）：`handle_action` 中待批量处理的智能体集合。
            # 变量说明（i）：当前序列或网格遍历使用的索引。
            human_agents = [i for i, agent in enumerate(self.agents) 
                          if hasattr(agent, 'is_human') and agent.is_human]
            if human_agents:
                if self.current_human_agent is None:
                    self.current_human_agent = human_agents[0]
                else:
                    # 变量说明（current_idx）：`handle_action` 中当前对象的唯一编号或序列索引。
                    current_idx = human_agents.index(self.current_human_agent)
                    self.current_human_agent = human_agents[(current_idx + 1) % len(human_agents)]
                print(f"Switched to Human Agent {self.current_human_agent}")
        elif action == 'human_action':
            self.start_human_action_selection()
    
    def start_human_action_selection(self):
        """Start human agent action selection process"""
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（human_agents）：`start_human_action_selection` 中待批量处理的智能体集合。
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        human_agents = [i for i, agent in enumerate(self.agents) 
                       if hasattr(agent, 'is_human') and agent.is_human]
        
        if not human_agents:
            print("No human agents available")
            return
        
        if self.current_human_agent is None:
            self.current_human_agent = human_agents[0]
        
        # Start input sequence
        self.input_mode = True
        self.input_type = 'op'
        self.input_prompt = f"Human Agent {self.current_human_agent} - Enter operation type (Navigator/share/noop/etc):"
        self.input_field = ""
    
    def process_input(self):
        """Process the current input"""
        if self.input_type == 'op':
            # 变量说明（op）：LLM 选定的高层操作名称，例如导航、制造或共享。
            op = self.input_field.strip()
            if not op:
                op = 'noop'
            
            # Store operation and move to next input
            self.agents[self.current_human_agent].op = op
            self.input_type = 'collect'
            self.input_prompt = f"Enter resource to collect (or 'not_applicable'):"
            self.input_field = ""
            
        elif self.input_type == 'collect':
            # 变量说明（rss_to_collect）：导航操作要寻找并采集的目标资源名称。
            rss_to_collect = self.input_field.strip()
            if not rss_to_collect:
                rss_to_collect = 'not_applicable'
            
            # Store collect resource and move to next input
            self.agents[self.current_human_agent].rss_to_collect = rss_to_collect
            self.input_type = 'share'
            self.input_prompt = f"Enter resource to share (or 'not_applicable'):"
            self.input_field = ""
            
        elif self.input_type == 'share':
            # 变量说明（rss_to_share）：共享操作准备转交给其他智能体的资源名称。
            rss_to_share = self.input_field.strip()
            if not rss_to_share:
                rss_to_share = 'not_applicable'
            
            # Store share resource and move to next input
            self.agents[self.current_human_agent].rss_to_share = rss_to_share
            self.input_type = 'target'
            self.input_prompt = f"Enter target agent id (or -1 if not applicable):"
            self.input_field = ""
            
        elif self.input_type == 'target':
            try:
                # 变量说明（target_agent_id）：资源共享操作的目标智能体编号；-1 表示无目标。
                target_agent_id = int(self.input_field.strip())
            except ValueError:
                target_agent_id = -1
            
            # Store target agent and complete input
            self.agents[self.current_human_agent].target_agent_id = target_agent_id
            
            # Update agent skills
            # 变量说明（agent）：当前正在处理的单个智能体对象。
            agent = self.agents[self.current_human_agent]
            agent.update_current_skill(agent.op, agent.rss_to_collect, agent.rss_to_share, target_agent_id)
            
            print(f"[Human Agent {self.current_human_agent}] Action set: op={agent.op}, collect={agent.rss_to_collect}, share={agent.rss_to_share}, target_agent_id={target_agent_id}")
            
            # Exit input mode
            self.input_mode = False
            self.input_field = ""
            self.input_type = ""
    
    def run_simulation_step(self):
        """Run one step of the simulation"""
        if self.paused or self.input_mode:
            return
        
        # Process agent actions
        self.action_processor.process_all_agent_actions(self.agents, self.env, self.n_players)
        
        # Collect actions
        # 变量说明（agents_actions）：所有智能体在当前时间步提交给环境的动作编号列表。
        agents_actions = self.action_processor.collect_agent_actions(self.agents, self.n_players)
        
        # Step environment
        # 变量说明（done）：本轮是否因采到钻石、超时或全部死亡而结束。
        # 变量说明（info）：环境附加状态，包含背包、成就、语义地图和生存状态。
        # 变量说明（obs）：环境返回的当前视觉观察；多人模式下按智能体编号排列。
        # 变量说明（rewards）：本时间步每个智能体获得的奖励值。
        obs, rewards, done, info = self.env_manager.step_environment(self.env, agents_actions)
        self.env_manager.update_crafting_stations(self.agents, self.env)
        
        # Update agent states
        self.agent_state_manager.update_all_agent_states(
            self.agents, obs, self.current_step, self.env, info, episode_number=0
        )
        
        # Process agent thinking
        # 变量说明（agents_with_new_thought）：按编号标记本步哪些智能体完成了重新规划。
        agents_with_new_thought = self.agent_state_manager.identify_agents_needing_thought(self.agents, info)
        # 变量说明（agents_contexts）：以智能体编号为键的个性化决策上下文字典。
        agents_contexts = self.simulation_context_manager.create_agent_contexts(self.agents, info)
        # 变量说明（agents_responses）：多个智能体并行思考后返回的结构化结果列表。
        agents_responses = self.agent_thinking_processor.process_agent_thinking_parallel(
            self.agents, agents_contexts, info
        )
        self.agent_thinking_processor.update_agents_from_responses(self.agents, agents_responses)
        
        # Show step report
        self.reporter.show_step_report(self.agents, agents_with_new_thought)
        
        self.current_step += 1
        
        # Check if simulation is done
        if done or self.current_step >= self.max_steps:
            print("Simulation completed!")
            self.running = False
    
    def run(self):
        """Main simulation loop"""
        print(f"Starting Fixed HumanAgent GUI with {self.n_players} agents")
        print(f"Human agents: {self.human_agent_ids}")
        print("Press 'h' to select human agent actions, 'p' to pause/resume, ESC to quit")
        
        while self.running:
            self.handle_events()
            self.run_simulation_step()
            self.render()
        
        pygame.quit()
        print("Fixed GUI closed")


def main():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = argparse.ArgumentParser(description='Fixed HumanAgent GUI for multi-agent simulation')
    parser.add_argument('--human_agents', type=str, default='',
                        help='Comma-separated list of agent ids to be human-controlled, e.g. "0,2"')
    parser.add_argument('--agent_num', type=int, default=3,
                        help='Number of agents (default: 3)')
    parser.add_argument('--step_num', type=int, default=350,
                        help='Number of steps (default: 350)')
    parser.add_argument('--fps', type=int, default=5,
                        help='FPS for GUI (default: 5)')
    
    # 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
    args = parser.parse_args()
    
    # Parse human agent IDs
    # 变量说明（human_agent_ids）：由用户手动控制的智能体编号集合。
    human_agent_ids = []
    if args.human_agents.strip():
        # 变量说明（x）：世界网格或图像中的横向坐标。
        human_agent_ids = [int(x.strip()) for x in args.human_agents.split(',') if x.strip().isdigit()]
    
    # Create and run GUI
    # 变量说明（gui）：负责窗口事件、布局和渲染的图形界面控制器。
    gui = FixedHumanAgentGUI(
        human_agent_ids=human_agent_ids,
        n_players=args.agent_num,
        max_steps=args.step_num
    )
    gui.fps = args.fps
    gui.run()


if __name__ == '__main__':
    main() 
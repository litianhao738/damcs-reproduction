#!/usr/bin/env python3
# 模块说明：多视角图形界面入口，用于观察或人工控制多个智能体。
"""
Multi-view GUI for HumanAgent integration with perspective switching
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


class MultiViewGUI:
    """Multi-view GUI class with perspective switching"""
    
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
        # 变量说明（self.input_prompt）：GUI 提示用户输入当前字段的说明文本。
        self.input_prompt = ""
        # 变量说明（self.input_field）：GUI 当前已经输入、尚未提交的字符串。
        self.input_field = ""
        # 变量说明（self.input_type）：人工输入字段需要转换成的目标数据类型。
        self.input_type = ""
        
        # View state
        # 变量说明（self.current_view_mode）：`__init__` 中当前处理的视图。
        self.current_view_mode = "single"  # "single", "multi", "overview"
        # 变量说明（self.current_followed_agent）：`__init__` 中当前处理的智能体。
        self.current_followed_agent = 0  # Which agent to follow in single view
        # 变量说明（self.view_layout）：`__init__` 中当前处理的视图。
        self.view_layout = "horizontal"  # "horizontal", "vertical", "grid"
        
        # Pygame setup
        # 变量说明（self.window_size）：GUI 主窗口的宽高像素值。
        self.window_size = (1400, 900)
        # 变量说明（self.game_area_size）：窗口中分配给游戏画面的宽高像素值。
        self.game_area_size = (600, 600)
        # 变量说明（self.sidebar_width）：`__init__` 中当前处理的宽度。
        self.sidebar_width = 400
        # 变量说明（self.fps）：界面刷新或视频编码的每秒帧数。
        self.fps = 5
        pygame.init()
        # 变量说明（self.screen）：Pygame 创建的主显示窗口表面。
        self.screen = pygame.display.set_mode(self.window_size)
        # 变量说明（self.clock）：控制 GUI 刷新频率的 Pygame 时钟。
        self.clock = pygame.time.Clock()
        
        # Colors
        # 变量说明（self.colors）：`__init__` 中待批量处理的颜色集合。
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'gray': (128, 128, 128),
            'dark_gray': (64, 64, 64),
            'light_gray': (192, 192, 192),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'cyan': (0, 255, 255)
        }
        
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
        }
        
        # UI buttons
        # 变量说明（self.buttons）：当前界面需要显示和响应的按钮描述列表。
        self.buttons = self.create_buttons()
        
        print('Multi-View GUI Controls:')
        print('  h: Human agent action selection')
        print('  p: Pause/Resume simulation')
        print('  v: Switch view mode (single/multi/overview)')
        print('  l: Switch layout (horizontal/vertical/grid)')
        print('  0-9: Follow specific agent (in single view)')
        print('  ESC: Quit')
        print('  Mouse: Click buttons for actions')
    
    def create_buttons(self):
        """Create UI buttons"""
        # 变量说明（buttons）：当前界面需要显示和响应的按钮描述列表。
        buttons = {}
        
        # View control buttons
        # 变量说明（view_buttons）：`create_buttons` 中待批量处理的视图集合。
        view_buttons = [
            ('Single View', 'single_view', 610, 650),
            ('Multi View', 'multi_view', 710, 650),
            ('Overview', 'overview', 810, 650),
            ('Layout', 'switch_layout', 910, 650)
        ]
        
        # 变量说明（action）：当前智能体准备执行的环境动作。
        # 变量说明（text）：`create_buttons` 中当前处理的文本。
        # 变量说明（x）：世界网格或图像中的横向坐标。
        # 变量说明（y）：世界网格或图像中的纵向坐标。
        for text, action, x, y in view_buttons:
            buttons[action] = {
                'rect': pygame.Rect(x, y, 80, 30),
                'text': text,
                'action': action,
                'color': self.colors['blue'],
                'hover_color': self.colors['orange']
            }
        
        # Agent follow buttons
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        for i in range(min(self.n_players, 10)):
            x = 610 + (i % 5) * 60
            y = 700 + (i // 5) * 35
            buttons[f'follow_agent_{i}'] = {
                'rect': pygame.Rect(x, y, 50, 25),
                'text': f'Agent {i}',
                'action': f'follow_agent_{i}',
                'color': self.colors['light_gray'],
                'hover_color': self.colors['yellow']
            }
        
        # Control buttons
        # 变量说明（control_buttons）：用于暂停、切换视图等仿真控制操作的按钮列表。
        control_buttons = [
            ('Pause', 'pause', 610, 750),
            ('Human Action', 'human_action', 700, 750),
            ('Help', 'help', 790, 750)
        ]
        
        for text, action, x, y in control_buttons:
            buttons[action] = {
                'rect': pygame.Rect(x, y, 80, 30),
                'text': text,
                'action': action,
                'color': self.colors['blue'],
                'hover_color': self.colors['orange']
            }
        
        return buttons
    
    def render(self):
        """Render the game state"""
        # Clear screen
        self.screen.fill(self.colors['black'])
        
        # Render based on view mode
        if self.current_view_mode == "single":
            self.render_single_view()
        elif self.current_view_mode == "multi":
            self.render_multi_view()
        elif self.current_view_mode == "overview":
            self.render_overview()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render buttons
        self.render_buttons()
        
        # Render input interface if in input mode
        if self.input_mode:
            self.render_input_interface()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def render_single_view(self):
        """Render single agent view"""
        # Switch to followed agent's perspective
        self.env.switch_player(self.current_followed_agent)
        
        # Render environment from that agent's perspective
        # 变量说明（image）：当前读取、处理或渲染的图像数组。
        image = self.env.render(self.game_area_size)
        # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render game area border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (0, 0, self.game_area_size[0], self.game_area_size[1]), 2)
        
        # Render current agent indicator
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 36)
        # 变量说明（agent_text）：`render_single_view` 中当前处理的智能体、文本。
        agent_text = font.render(f'Following Agent {self.current_followed_agent}', True, self.colors['yellow'])
        self.screen.blit(agent_text, (10, self.game_area_size[1] + 10))
    
    def render_multi_view(self):
        """Render multiple agent views"""
        # Get all agent views
        # 变量说明（all_views）：`render_multi_view` 中待批量处理的视图集合。
        all_views = self.env.render_all(self.game_area_size)
        
        if self.view_layout == "horizontal":
            self.render_horizontal_layout(all_views)
        elif self.view_layout == "vertical":
            self.render_vertical_layout(all_views)
        elif self.view_layout == "grid":
            self.render_grid_layout(all_views)
    
    def render_horizontal_layout(self, all_views):
        # 变量说明（all_views）：`render_horizontal_layout` 中待批量处理的视图集合。
        """Render views in horizontal layout"""
        # 变量说明（view_width）：`render_horizontal_layout` 中当前处理的视图、宽度。
        view_width = self.window_size[0] // len(all_views)
        # 变量说明（view_height）：`render_horizontal_layout` 中当前处理的视图、高度。
        view_height = self.game_area_size[1]
        
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        # 变量说明（view）：单个玩家局部观察窗口的行列尺寸。
        for i, view in enumerate(all_views):
            # 变量说明（x）：世界网格或图像中的横向坐标。
            x = i * view_width
            # 变量说明（y）：世界网格或图像中的纵向坐标。
            y = 0
            
            # Convert view to surface
            # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
            font = pygame.font.Font(None, 24)
            # 变量说明（agent_text）：`render_horizontal_layout` 中当前处理的智能体、文本。
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_vertical_layout(self, all_views):
        # 变量说明（all_views）：`render_vertical_layout` 中待批量处理的视图集合。
        """Render views in vertical layout"""
        # 变量说明（view_width）：`render_vertical_layout` 中当前处理的视图、宽度。
        view_width = self.game_area_size[0]
        # 变量说明（view_height）：`render_vertical_layout` 中当前处理的视图、高度。
        view_height = self.window_size[1] // len(all_views)
        
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        # 变量说明（view）：单个玩家局部观察窗口的行列尺寸。
        for i, view in enumerate(all_views):
            # 变量说明（x）：世界网格或图像中的横向坐标。
            x = 0
            # 变量说明（y）：世界网格或图像中的纵向坐标。
            y = i * view_height
            
            # Convert view to surface
            # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
            font = pygame.font.Font(None, 24)
            # 变量说明（agent_text）：`render_vertical_layout` 中当前处理的智能体、文本。
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_grid_layout(self, all_views):
        # 变量说明（all_views）：`render_grid_layout` 中待批量处理的视图集合。
        """Render views in grid layout"""
        # 变量说明（cols）：网格布局需要划分的列数。
        cols = int(np.ceil(np.sqrt(len(all_views))))
        # 变量说明（rows）：`render_grid_layout` 中待批量处理的行数集合。
        rows = int(np.ceil(len(all_views) / cols))
        
        # 变量说明（view_width）：`render_grid_layout` 中当前处理的视图、宽度。
        view_width = self.window_size[0] // cols
        # 变量说明（view_height）：`render_grid_layout` 中当前处理的视图、高度。
        view_height = self.window_size[1] // rows
        
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        # 变量说明（view）：单个玩家局部观察窗口的行列尺寸。
        for i, view in enumerate(all_views):
            # 变量说明（col）：当前处理的网格列索引。
            col = i % cols
            # 变量说明（row）：`render_grid_layout` 中当前处理的行数。
            row = i // cols
            # 变量说明（x）：世界网格或图像中的横向坐标。
            x = col * view_width
            # 变量说明（y）：世界网格或图像中的纵向坐标。
            y = row * view_height
            
            # Convert view to surface
            # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
            surface = pygame.surfarray.make_surface(view.transpose((1, 0, 2)))
            self.screen.blit(surface, (x, y))
            
            # Render border
            pygame.draw.rect(self.screen, self.colors['white'], 
                           (x, y, view_width, view_height), 2)
            
            # Render agent label
            # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
            font = pygame.font.Font(None, 20)
            # 变量说明（agent_text）：`render_grid_layout` 中当前处理的智能体、文本。
            agent_text = font.render(f'Agent {i}', True, self.colors['yellow'])
            self.screen.blit(agent_text, (x + 5, y + 5))
    
    def render_overview(self):
        """Render overview of all agents"""
        # Use the first agent's view as overview
        self.env.switch_player(0)
        # 变量说明（image）：当前读取、处理或渲染的图像数组。
        image = self.env.render(self.game_area_size)
        # 变量说明（surface）：由观察图像转换得到、准备绘制到窗口的 Pygame Surface。
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        
        # Render game area border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (0, 0, self.game_area_size[0], self.game_area_size[1]), 2)
        
        # Render overview label
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 36)
        # 变量说明（overview_text）：`render_overview` 中当前处理的文本。
        overview_text = font.render('Overview (Agent 0 Perspective)', True, self.colors['cyan'])
        self.screen.blit(overview_text, (10, self.game_area_size[1] + 10))
    
    def render_sidebar(self):
        """Render the sidebar with agent information"""
        # 变量说明（sidebar_x）：GUI 右侧控制栏起始位置的横坐标。
        sidebar_x = self.window_size[0] - self.sidebar_width + 10
        
        # Title
        # 变量说明（font_large）：GUI 标题或重要状态文字使用的大号 Pygame 字体。
        font_large = pygame.font.Font(None, 36)
        # 变量说明（title）：当前窗口、图表或输出区域显示的标题文本。
        title = font_large.render('Multi-View GUI', True, self.colors['white'])
        self.screen.blit(title, (sidebar_x, 10))
        
        # View mode info
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 24)
        # 变量说明（view_text）：`render_sidebar` 中当前处理的视图、文本。
        view_text = font.render(f'View Mode: {self.current_view_mode}', True, self.colors['cyan'])
        self.screen.blit(view_text, (sidebar_x, 50))
        
        if self.current_view_mode == "single":
            # 变量说明（follow_text）：`render_sidebar` 中当前处理的文本。
            follow_text = font.render(f'Following: Agent {self.current_followed_agent}', True, self.colors['yellow'])
            self.screen.blit(follow_text, (sidebar_x, 80))
        
        # 变量说明（layout_text）：`render_sidebar` 中当前处理的文本。
        layout_text = font.render(f'Layout: {self.view_layout}', True, self.colors['cyan'])
        self.screen.blit(layout_text, (sidebar_x, 110))
        
        # Step counter
        # 变量说明（step_text）：`render_sidebar` 中当前处理的时间步、文本。
        step_text = font.render(f'Step: {self.current_step}/{self.max_steps}', True, self.colors['white'])
        self.screen.blit(step_text, (sidebar_x, 140))
        
        # Progress bar
        # 变量说明（progress）：当前目标、下载或实验任务的完成进度。
        progress = self.current_step / self.max_steps
        # 变量说明（bar_width）：`render_sidebar` 中当前处理的宽度。
        bar_width = 300
        # 变量说明（bar_height）：`render_sidebar` 中当前处理的高度。
        bar_height = 20
        # 变量说明（bar_x）：状态条在画布上的左侧横坐标。
        bar_x = sidebar_x
        # 变量说明（bar_y）：状态条在画布上的顶部纵坐标。
        bar_y = 170
        
        # Background
        pygame.draw.rect(self.screen, self.colors['dark_gray'], 
                        (bar_x, bar_y, bar_width, bar_height))
        # Progress
        pygame.draw.rect(self.screen, self.colors['green'], 
                        (bar_x, bar_y, int(bar_width * progress), bar_height))
        # Border
        pygame.draw.rect(self.screen, self.colors['white'], 
                        (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Agent status
        self.render_agent_status(sidebar_x, bar_y + 40)
        
        # Status indicators
        self.render_status_indicators(sidebar_x, 500)
    
    def render_agent_status(self, x, y):
        # 变量说明（x）：世界网格或图像中的横向坐标。
        # 变量说明（y）：世界网格或图像中的纵向坐标。
        """Render agent status information"""
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 20)
        # 变量说明（title）：当前窗口、图表或输出区域显示的标题文本。
        title = font.render('Agent Status:', True, self.colors['yellow'])
        self.screen.blit(title, (x, y))
        
        # 变量说明（y_offset）：当前元素相对布局起点的纵向像素偏移量。
        y_offset = y + 30
        # 变量说明（agent）：当前正在处理的单个智能体对象。
        # 变量说明（i）：当前序列或网格遍历使用的索引。
        for i, agent in enumerate(self.agents):
            # Agent type and status
            # 变量说明（agent_type）：`render_agent_status` 中当前处理的智能体。
            agent_type = "Human" if hasattr(agent, 'is_human') and agent.is_human else "AI"
            # 变量说明（status）：调用方准备写入对象的最新执行状态。
            status = "Waiting" if agent.action_status == ActionStatus.DONE else "Working"
            # 变量说明（color）：`render_agent_status` 中当前处理的颜色。
            color = self.colors['yellow'] if hasattr(agent, 'is_human') and agent.is_human else self.colors['white']
            
            # Highlight current followed agent
            if self.current_view_mode == "single" and i == self.current_followed_agent:
                color = self.colors['cyan']
            
            # 变量说明（agent_text）：`render_agent_status` 中当前处理的智能体、文本。
            agent_text = font.render(f'Agent {i} ({agent_type}): {status}', True, color)
            self.screen.blit(agent_text, (x, y_offset))
            
            # Current operation
            if hasattr(agent, 'op') and agent.op:
                # 变量说明（op_text）：`render_agent_status` 中当前处理的文本。
                op_text = font.render(f'  Op: {agent.op}', True, self.colors['light_gray'])
                self.screen.blit(op_text, (x + 10, y_offset + 20))
            
            y_offset += 50
    
    def render_status_indicators(self, x, y):
        # 变量说明（x）：世界网格或图像中的横向坐标。
        # 变量说明（y）：世界网格或图像中的纵向坐标。
        """Render status indicators"""
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 20)
        
        # Pause indicator
        if self.paused:
            # 变量说明（pause_text）：`render_status_indicators` 中当前处理的文本。
            pause_text = font.render('PAUSED - Press P to resume', True, self.colors['red'])
            self.screen.blit(pause_text, (x, y))
        
        # Input mode indicator
        if self.input_mode:
            # 变量说明（input_text）：`render_status_indicators` 中当前处理的文本。
            input_text = font.render('INPUT MODE - Type your response', True, self.colors['green'])
            self.screen.blit(input_text, (x, y + 25))
        
        # Current human agent
        if self.current_human_agent is not None:
            # 变量说明（current_text）：`render_status_indicators` 中当前处理的文本。
            current_text = font.render(f'Current Human Agent: {self.current_human_agent}', True, self.colors['orange'])
            self.screen.blit(current_text, (x, y + 50))
    
    def render_buttons(self):
        """Render UI buttons"""
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 18)
        # 变量说明（mouse_pos）：当前鼠标指针在 GUI 窗口内的像素坐标。
        mouse_pos = pygame.mouse.get_pos()
        
        # 变量说明（button）：当前绘制或处理点击事件的界面按钮描述。
        # 变量说明（button_id）：`render_buttons` 中当前对象的唯一编号或序列索引。
        for button_id, button in self.buttons.items():
            # Check hover
            # 变量说明（color）：`render_buttons` 中当前处理的颜色。
            color = button['hover_color'] if button['rect'].collidepoint(mouse_pos) else button['color']
            
            # Highlight current view mode
            if button_id == f'{self.current_view_mode}_view':
                color = self.colors['green']
            
            # Highlight current followed agent
            if button_id == f'follow_agent_{self.current_followed_agent}':
                color = self.colors['green']
            
            # Draw button
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['white'], button['rect'], 2)
            
            # Draw text
            # 变量说明（text）：`render_buttons` 中当前处理的文本。
            text = font.render(button['text'], True, self.colors['black'])
            # 变量说明（text_rect）：`render_buttons` 中当前处理的文本。
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
    
    def render_input_interface(self):
        """Render input interface overlay"""
        # Semi-transparent overlay
        # 变量说明（overlay）：叠加到游戏画面上、用于增强可读性的半透明图层。
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(128)
        overlay.fill(self.colors['black'])
        self.screen.blit(overlay, (0, 0))
        
        # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
        font = pygame.font.Font(None, 24)
        
        # Input prompt
        # 变量说明（prompt_text）：`render_input_interface` 中当前处理的文本。
        prompt_text = font.render(self.input_prompt, True, self.colors['white'])
        self.screen.blit(prompt_text, (50, 200))
        
        # Input field
        # 变量说明（input_text）：`render_input_interface` 中当前处理的文本。
        input_text = font.render(f'Input: {self.input_field}', True, self.colors['yellow'])
        self.screen.blit(input_text, (50, 230))
        
        # Instructions
        # 变量说明（instruction_text）：`render_input_interface` 中当前处理的文本。
        instruction_text = font.render('Press ENTER to confirm, ESC to cancel', True, self.colors['light_gray'])
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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_mouse_click(event.pos)
    
    def handle_normal_event(self, event):
        # 变量说明（event）：`handle_normal_event` 中当前处理的事件。
        """Handle events when not in input mode"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_h:
            self.start_human_action_selection()
        elif event.key == pygame.K_v:
            self.switch_view_mode()
        elif event.key == pygame.K_l:
            self.switch_layout()
        elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                          pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
            # Follow specific agent
            # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
            agent_id = event.key - pygame.K_0
            if agent_id < self.n_players:
                self.current_followed_agent = agent_id
                print(f"Now following Agent {agent_id}")
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
    
    def handle_mouse_click(self, pos):
        # 变量说明（pos）：对象在世界网格中的二维坐标。
        """Handle mouse clicks on buttons"""
        # 变量说明（button）：当前绘制或处理点击事件的界面按钮描述。
        # 变量说明（button_id）：`handle_mouse_click` 中当前对象的唯一编号或序列索引。
        for button_id, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                self.handle_button_action(button_id)
                break
    
    def handle_button_action(self, button_id):
        # 变量说明（button_id）：`handle_button_action` 中当前对象的唯一编号或序列索引。
        """Handle button actions"""
        if button_id == 'pause':
            self.paused = not self.paused
        elif button_id == 'human_action':
            self.start_human_action_selection()
        elif button_id == 'single_view':
            self.current_view_mode = "single"
        elif button_id == 'multi_view':
            self.current_view_mode = "multi"
        elif button_id == 'overview':
            self.current_view_mode = "overview"
        elif button_id == 'switch_layout':
            self.switch_layout()
        elif button_id.startswith('follow_agent_'):
            # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
            agent_id = int(button_id.split('_')[-1])
            if agent_id < self.n_players:
                self.current_followed_agent = agent_id
                self.current_view_mode = "single"
                print(f"Now following Agent {agent_id}")
    
    def switch_view_mode(self):
        """Switch between view modes"""
        # 变量说明（modes）：冒烟测试支持的聊天、嵌入等运行模式列表。
        modes = ["single", "multi", "overview"]
        # 变量说明（current_index）：`switch_view_mode` 中当前对象的唯一编号或序列索引。
        current_index = modes.index(self.current_view_mode)
        self.current_view_mode = modes[(current_index + 1) % len(modes)]
        print(f"Switched to {self.current_view_mode} view mode")
    
    def switch_layout(self):
        """Switch between multi-view layouts"""
        # 变量说明（layouts）：多视角 GUI 支持的布局模式名称列表。
        layouts = ["horizontal", "vertical", "grid"]
        # 变量说明（current_index）：`switch_layout` 中当前对象的唯一编号或序列索引。
        current_index = layouts.index(self.view_layout)
        self.view_layout = layouts[(current_index + 1) % len(layouts)]
        print(f"Switched to {self.view_layout} layout")
    
    def handle_action(self, action):
        # 变量说明（action）：当前智能体准备执行的环境动作。
        """Handle game actions"""
        print(f"Action selected: {action}")
    
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
        print(f"Starting Multi-View GUI with {self.n_players} agents")
        print(f"Human agents: {self.human_agent_ids}")
        print("Press 'h' for human actions, 'p' to pause/resume, 'v' to switch view, 'l' to switch layout")
        print("Press 0-9 to follow specific agent, ESC to quit")
        
        while self.running:
            self.handle_events()
            self.run_simulation_step()
            self.render()
        
        pygame.quit()
        print("Multi-View GUI closed")


def main():
    # 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
    parser = argparse.ArgumentParser(description='Multi-View GUI for HumanAgent simulation')
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
    gui = MultiViewGUI(
        human_agent_ids=human_agent_ids,
        n_players=args.agent_num,
        max_steps=args.step_num
    )
    gui.fps = args.fps
    gui.run()


if __name__ == '__main__':
    main() 
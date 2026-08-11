# 模块说明：将实验保存的逐帧图像合成为单视角或多面板视频。
import os
import cv2
from PIL import Image
import numpy as np
import argparse
import math
from PIL import ImageDraw, ImageFont

# Parse command line arguments
# 变量说明（parser）：命令行参数解析器，用于注册并读取启动选项。
parser = argparse.ArgumentParser(description='Combine agent step images into videos for multiple agents.')
parser.add_argument('--results_dir', type=str, default='results', help='Path to results folder containing agent_x subfolders')
parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output videos')
parser.add_argument('--fps', type=int, default=5, help='Frames per second for the video')
parser.add_argument('--agents', type=str, default=None, help='Comma-separated list of agent ids to process (e.g., 0,2,3). If not set, process all agent folders.')
parser.add_argument('--multi_panel', action='store_true', help='If set, create a multi-agent panel video instead of individual videos')
parser.add_argument('--output', type=str, default='multi_agent_panel.mp4', help='Output filename for multi-panel video (only used if --multi_panel)')
# 变量说明（args）：命令行解析后的参数对象，集中保存本次运行配置。
args = parser.parse_args()

# Find agent folders
def get_agent_folders(results_dir, agent_ids=None):
    # 变量说明（results_dir）：保存统计表、观察帧和其他实验产物的目录。
    # 变量说明（agent_ids）：`get_agent_folders` 中待批量处理的智能体集合。
    # 变量说明（folders）：结果目录中符合智能体帧目录命名规则的文件夹列表。
    folders = []
    # 变量说明（name）：当前对象、指标或配置项的名称。
    for name in os.listdir(results_dir):
        if name.startswith('agent_') and os.path.isdir(os.path.join(results_dir, name)):
            # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
            agent_id = name.split('_')[-1]
            if agent_ids is None or agent_id in agent_ids:
                folders.append((agent_id, os.path.join(results_dir, name)))
    # 变量说明（x）：世界网格或图像中的横向坐标。
    return sorted(folders, key=lambda x: int(x[0]))

# Parse agent ids if provided
# 变量说明（agent_ids）：模块 中待批量处理的智能体集合。
agent_ids = None
if args.agents:
    agent_ids = set(args.agents.split(','))

# 变量说明（agent_folders）：模块 中待批量处理的智能体集合。
agent_folders = get_agent_folders(args.results_dir, agent_ids)
if not agent_folders:
    raise RuntimeError(f'No agent folders found in {args.results_dir}')

os.makedirs(args.output_dir, exist_ok=True)

def make_video_for_agent(agent_id, agent_dir, output_path, fps):
    # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
    # 变量说明（agent_dir）：`make_video_for_agent` 保存或读取相关文件的目录。
    # 变量说明（output_path）：当前输出文件的完整路径。
    # 变量说明（fps）：界面刷新或视频编码的每秒帧数。
    # 变量说明（f）：当前循环使用的文件、函数或 A* 估计总代价值。
    # 变量说明（image_files）：`make_video_for_agent` 中待批量处理的图像、文件集合。
    image_files = [f for f in os.listdir(agent_dir) if f.endswith('.png') and not f.endswith('_stats.png')]
    # 变量说明（x）：世界网格或图像中的横向坐标。
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
    if not image_files:
        print(f'No step images found in {agent_dir}, skipping.')
        return
    # 变量说明（first_img）：用于确定视频尺寸和通道格式的第一帧图像。
    first_img = Image.open(os.path.join(agent_dir, image_files[0]))
    # 变量说明（frame_size）：`make_video_for_agent` 中当前处理的图像帧。
    frame_size = first_img.size  # (width, height)
    # 变量说明（fourcc）：OpenCV 视频编码器使用的四字符编码标识。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 变量说明（video_writer）：`make_video_for_agent` 中当前处理的视频。
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    # 变量说明（fname）：当前遍历到的文件名称。
    for fname in image_files:
        # 变量说明（img_path）：`make_video_for_agent` 读取或写入当前文件时使用的完整路径。
        img_path = os.path.join(agent_dir, fname)
        # 变量说明（img）：当前加载或正在处理的图像数组。
        img = Image.open(img_path).convert('RGB')
        # 变量说明（img_np）：从 Pillow 图像转换得到的 NumPy 像素数组。
        img_np = np.array(img)
        # 变量说明（img_cv）：转换为 OpenCV BGR 通道顺序的图像数组。
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if (img_cv.shape[1], img_cv.shape[0]) != frame_size:
            img_cv = cv2.resize(img_cv, frame_size)
        video_writer.write(img_cv)
    video_writer.release()
    print(f'Agent {agent_id}: Video saved as {output_path}')

def add_label_to_image(img, label, font=None, label_height=20):
    # 变量说明（img）：当前加载或正在处理的图像数组。
    # 变量说明（label）：`add_label_to_image` 中当前处理的标签。
    # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
    # 变量说明（label_height）：`add_label_to_image` 中当前处理的标签、高度。
    # 变量说明（h）：当前区域、图像或纹理的像素高度。
    # 变量说明（w）：当前区域、图像或纹理的像素宽度。
    w, h = img.size
    # 变量说明（new_img）：完成尺寸统一、标签绘制或颜色转换后的新图像。
    new_img = Image.new('RGB', (w, h + label_height), color=(0,0,0))
    new_img.paste(img, (0, 0))
    # 变量说明（draw）：用于在图像上绘制形状和文本的 Pillow 绘图对象。
    draw = ImageDraw.Draw(new_img)
    if font is None:
        font = ImageFont.load_default()
    # Use textbbox for accurate text size
    # 变量说明（bbox）：当前对象或视图在坐标系中的边界矩形。
    bbox = draw.textbbox((0, 0), label, font=font)
    # 变量说明（text_w）：`add_label_to_image` 中当前处理的文本。
    text_w = bbox[2] - bbox[0]
    # 变量说明（text_h）：`add_label_to_image` 中当前处理的文本。
    text_h = bbox[3] - bbox[1]
    # 变量说明（text_x）：`add_label_to_image` 中当前处理的文本。
    text_x = (w - text_w) // 2
    # 变量说明（text_y）：`add_label_to_image` 中当前处理的文本。
    text_y = h + (label_height - text_h) // 2
    draw.text((text_x, text_y), label, fill=(255,255,255), font=font)
    return new_img

def make_multi_panel_video(agent_folders, output_path, fps):
    # Collect sorted image lists for each agent
    # 变量说明（agent_folders）：`make_multi_panel_video` 中待批量处理的智能体集合。
    # 变量说明（output_path）：当前输出文件的完整路径。
    # 变量说明（fps）：界面刷新或视频编码的每秒帧数。
    # 变量说明（all_image_lists）：`make_multi_panel_video` 中待批量处理的图像集合。
    all_image_lists = []
    # 变量说明（agent_labels）：`make_multi_panel_video` 中待批量处理的智能体、标签集合。
    agent_labels = []
    # 变量说明（agent_dir）：`make_multi_panel_video` 保存或读取相关文件的目录。
    # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
    for agent_id, agent_dir in agent_folders:
        # 变量说明（f）：当前循环使用的文件、函数或 A* 估计总代价值。
        # 变量说明（image_files）：`make_multi_panel_video` 中待批量处理的图像、文件集合。
        image_files = [f for f in os.listdir(agent_dir) if f.endswith('.png') and not f.endswith('_stats.png')]
        # 变量说明（x）：世界网格或图像中的横向坐标。
        image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
        if not image_files:
            print(f'No step images found in {agent_dir}, skipping agent {agent_id}.')
            continue
        all_image_lists.append([os.path.join(agent_dir, f) for f in image_files])
        agent_labels.append(f'Agent {agent_id}')
    if not all_image_lists:
        print('No agent images found for multi-panel video.')
        return
    # Use the minimum number of frames across all agents
    # 变量说明（lst）：多面板视频中当前智能体可用的帧路径列表。
    # 变量说明（min_frames）：`make_multi_panel_video` 中待批量处理的图像帧集合。
    min_frames = min(len(lst) for lst in all_image_lists)
    # 变量说明（n_agents）：`make_multi_panel_video` 中待批量处理的智能体集合。
    n_agents = len(all_image_lists)
    # Determine grid size (try to make it as square as possible)
    # 变量说明（grid_cols）：多视图网格布局计算出的列数。
    grid_cols = math.ceil(math.sqrt(n_agents))
    # 变量说明（grid_rows）：`make_multi_panel_video` 中待批量处理的行数集合。
    grid_rows = math.ceil(n_agents / grid_cols)
    # Get frame size from first image
    # 变量说明（first_img）：用于确定视频尺寸和通道格式的第一帧图像。
    first_img = Image.open(all_image_lists[0][0])
    # 变量说明（frame_h）：`make_multi_panel_video` 中当前处理的图像帧。
    # 变量说明（frame_w）：`make_multi_panel_video` 中当前处理的图像帧。
    frame_w, frame_h = first_img.size
    # 变量说明（label_height）：`make_multi_panel_video` 中当前处理的标签、高度。
    label_height = 20
    # 变量说明（panel_w）：多面板布局中单个子画面的像素宽度。
    panel_w = frame_w * grid_cols
    # 变量说明（panel_h）：多面板布局中单个子画面的像素高度。
    panel_h = (frame_h + label_height) * grid_rows
    # 变量说明（fourcc）：OpenCV 视频编码器使用的四字符编码标识。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 变量说明（video_writer）：`make_multi_panel_video` 中当前处理的视频。
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (panel_w, panel_h))
    # 变量说明（font）：用于绘制当前界面文字的 Pygame 字体对象。
    font = ImageFont.load_default()
    # 变量说明（i）：当前序列或网格遍历使用的索引。
    for i in range(min_frames):
        # 变量说明（panel）：多视角视频或 GUI 中当前智能体对应的子画面。
        panel = Image.new('RGB', (panel_w, panel_h), color=(0,0,0))
        # 变量说明（idx）：当前元素在序列中的零基索引。
        # 变量说明（img_list）：等待拼接成面板的智能体图像列表。
        for idx, img_list in enumerate(all_image_lists):
            # 变量说明（row）：`make_multi_panel_video` 中当前处理的行数。
            row = idx // grid_cols
            # 变量说明（col）：当前处理的网格列索引。
            col = idx % grid_cols
            # 变量说明（img）：当前加载或正在处理的图像数组。
            img = Image.open(img_list[i]).convert('RGB')
            # 变量说明（img_labeled）：已经绘制智能体编号标签的图像帧。
            img_labeled = add_label_to_image(img, agent_labels[idx], font=font, label_height=label_height)
            panel.paste(img_labeled, (col * frame_w, row * (frame_h + label_height)))
        # 变量说明（panel_np）：由 Pillow 面板转换得到的 NumPy RGB 图像数组。
        panel_np = np.array(panel)
        # 变量说明（panel_cv）：转换为 OpenCV BGR 格式、准备写入视频的多视角面板。
        panel_cv = cv2.cvtColor(panel_np, cv2.COLOR_RGB2BGR)
        video_writer.write(panel_cv)
    video_writer.release()
    print(f'Multi-agent panel video saved as {output_path}')

if args.multi_panel:
    make_multi_panel_video(agent_folders, args.output, args.fps)
else:
    # 变量说明（agent_dir）：模块 保存或读取相关文件的目录。
    # 变量说明（agent_id）：智能体在多人环境中的唯一编号。
    for agent_id, agent_dir in agent_folders:
        # 变量说明（output_path）：当前输出文件的完整路径。
        output_path = os.path.join(args.output_dir, f'agent_{agent_id}.mp4')
        make_video_for_agent(agent_id, agent_dir, output_path, args.fps) 
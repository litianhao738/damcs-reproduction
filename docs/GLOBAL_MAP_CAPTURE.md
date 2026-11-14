# 全局地图截图功能：修改记录

## 目的

为 DAMCS 多智能体 Crafter 实验保存矿石采集末阶段的**完整 64×64 地图**。此前 `results/agent_<id>/<step>.png` 是某一智能体的局部观测和背包栏，无法还原全局资源、地形与所有玩家的位置。

## 修改内容

| 文件 | 修改 | 作用 |
| --- | --- | --- |
| `crafter/engine.py` | 实现 `GlobalView` | 绘制全部地形格，再叠加玩家、动物、怪物等对象，输出完整地图图像。 |
| `crafter/env.py` | 增加全局地图配置和 `save_global_map()` | 在每步环境状态更新、局部视图生成后，按条件保存全局地图 PNG。 |
| `run.py` | 将截图配置传入环境 | 保存目录统一放在本轮实验的 `results_dir/global_maps`。 |
| `utils.py` | 增加 3 个命令行参数 | 无需再修改源代码即可控制是否截图、起始步数和图片分辨率。 |
| `README.md` | 增加使用说明 | 记录实验命令、输出目录和复现注意事项。 |

## 新增命令行参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--save_global_maps` | 关闭 | 开启完整地图截图。 |
| `--global_map_start` | `0` | 从该环境步开始，每一步保存一张全局地图。 |
| `--global_map_unit` | `8` | 每个地图格对应的像素边长；64×64 地图使用 `8` 时输出 512×512 PNG。 |

## 本次复现实验建议

原实验在第 134 个环境步首次完成 `Collect diamond`。为了保留挖到钻石前后的演化过程，建议设置 `--global_map_start 130`，并将结果输出到一个新的目录：

```powershell
python run.py --agent_num 6 --step_num 150 --num_of_turn 1 --results_dir results\qwen37_plus_20260526_agent_6a_150s_with_global_map --save_global_maps --global_map_start 130 --global_map_unit 8 --no_plot
```

如果该次运行仍在第 134 步结束，将生成 `global_map_step_0130.png` 至 `global_map_step_0134.png`。LLM 推理可能存在随机性，因此新一轮的完成步数可能变化；该功能会保存这一次实际运行的末阶段地图。

当 `--num_of_turn` 大于 1 时，图片会分别保存到 `global_maps/round_1/`、`global_maps/round_2/` 等目录，避免不同轮次覆盖同名文件。

# 模块说明：保存 Crafter 动作、物品和成就名称常量。
import pathlib

import ruamel.yaml

# 变量说明（root）：资源目录或搜索结构使用的根节点。
root = pathlib.Path(__file__).parent
# 变量说明（yaml）：从 Crafter 数据文件加载的配置字典。
yaml = ruamel.yaml.YAML(typ='safe', pure=True)
# 变量说明（key）：当前字典条目的键。
# 变量说明（value）：当前配置项、字段或计算结果对应的值。
for key, value in yaml.load((root / 'data.yaml').read_text()).items():
  globals()[key] = value

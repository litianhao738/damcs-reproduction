"""Readable descriptions of progress options: Chinese for the review page, English for agent notes."""

ITEM_ZH = {
    "wood": "木头", "stone": "石头", "coal": "煤", "iron": "铁", "diamond": "钻石",
    "wood_pickaxe": "木镐", "stone_pickaxe": "石镐", "iron_pickaxe": "铁镐",
    "table": "工作台", "furnace": "熔炉", "tree": "树",
}
MILESTONE_ZH = {
    "collect_wood": "采木头", "place_table": "放工作台", "make_wood_pickaxe": "做木镐",
    "collect_stone": "采石头", "make_stone_pickaxe": "做石镐", "collect_coal": "采煤",
    "place_furnace": "放熔炉", "collect_iron": "采铁", "make_iron_pickaxe": "做铁镐",
    "collect_diamond": "采钻石",
}
TEMPLATE_ZH = {
    "supply": "补给", "substitute": "换人做", "build_here": "就地建", "gather": "改派采集", "push": "派人推箱",
}
PROBLEM_TYPE_ZH = {"exact": "这一步就能完成，但当前计划没有做", "slow": "明显比最好的方案慢", "unknown": "团队卡住，估不出方案"}


def milestone_zh(milestone):
    if milestone.startswith("deliver_block_"):
        return f"送达箱子 {milestone.rsplit('_', 1)[1]}"
    return MILESTONE_ZH.get(milestone, milestone)


def _item(name):
    return ITEM_ZH.get(name, name)


def macro_zh(macro):
    kind = macro[0]
    if kind == "share":
        _, item, recipient, count = macro
        return f"把 {count} 个{_item(item)}送给 Agent {recipient}"
    if kind == "collect":
        material, count = macro[1], macro[2]
        item = "木头" if material == "tree" else _item(material)
        return f"去采 {count} 个{item}"
    if kind == "make":
        return f"到设施旁做{_item(macro[1])}"
    if kind == "place":
        return f"就地放{_item(macro[1])}"
    if kind == "push":
        _, block_id, row = macro
        return f"站到箱子 {block_id} 左侧第 {row + 1} 格，人齐后一起向右推，直到送达"
    return str(macro)


def plan_zh(plan):
    operation = plan["operation"]
    if operation == "share":
        return f"把 1 个{_item(plan['item'])}送给 Agent {plan['recipient']}"
    if operation.startswith("make_"):
        return f"这一步做{_item(operation[5:])}"
    if operation.startswith("place_"):
        return f"这一步放{_item(operation[6:])}"
    if operation == "do":
        return f"这一步采{_item(plan['material'])}"
    if operation == "move_right":
        return "这一步向右推"
    return operation


def option_lines_zh(option):
    """One line per agent the option involves."""
    if option.get("tasks"):
        return [f"Agent {agent_id}：" + "，然后".join(macro_zh(macro) for macro in macros)
                for agent_id, macros in sorted(option["tasks"].items(), key=lambda entry: int(entry[0]))]
    return [f"Agent {agent_id}：{plan_zh(plan)}"
            for agent_id, plan in sorted(option.get("first_plans", {}).items(), key=lambda entry: int(entry[0]))]


def macro_en(macro):
    kind = macro[0]
    if kind == "share":
        return f"give {macro[3]} {macro[1]} to agent {macro[2]}"
    if kind == "collect":
        return f"collect {macro[2]} {macro[1]}"
    if kind == "make":
        return f"make {macro[1]} next to the facility"
    if kind == "place":
        return f"place a {macro[1]} here"
    if kind == "push":
        return f"stand at row {macro[2] + 1} of block {macro[1]}'s left face and push right with the others"
    return str(macro)


def option_en(option):
    """Short English summary, written into the agents' context as a team assignment."""
    if option.get("tasks"):
        parts = [f"agent {agent_id}: " + ", then ".join(macro_en(macro) for macro in macros)
                 for agent_id, macros in sorted(option["tasks"].items(), key=lambda entry: int(entry[0]))]
    else:
        parts = [f"agent {agent_id}: {plan['operation']}"
                 + (f" {plan['item']} to agent {plan['recipient']}" if plan["operation"] == "share" else "")
                 for agent_id, plan in sorted(option.get("first_plans", {}).items(), key=lambda entry: int(entry[0]))]
    return f"{option['milestone']} ({option['template']}, about {option['estimate']} steps): " + "; ".join(parts)

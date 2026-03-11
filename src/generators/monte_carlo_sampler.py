# -*- coding: utf-8 -*-
"""
蒙特卡洛采样器
在14种元素空间中随机采样二元、三元化合物
约束：总原子数不超过20
"""
import os
import random
import logging
import math
from datetime import datetime
from typing import Any

# 元素列表
ELEMENTS = ['Ti', 'V', 'Cu', 'Ag', 'In', 'Ge', 'Sn', 'Pb', 'As', 'Sb', 'Bi', 'S', 'Se', 'Te']

# 配置
MAX_ATOMS = 20  # 最大原子数
N_SAMPLES = 100  # 采样数量

# Default hard constraints: AxByChz
DEFAULT_HARD_CONSTRAINTS = {
    "schema": {
        "type": "A-B-Ch",
        "groups": {
            "A": ["Ag", "Cu", "In", "Sn", "Pb"],
            "B": ["As", "Sb", "Ge", "Bi", "Ti", "V"],
            "Ch": ["S", "Se", "Te"],
        },
    },
    "stoichiometry": {
        "A": {"min": 2, "max": 10},
        "B": {"min": 1, "max": 10},
        "Ch": {"min": 1, "max": 10},
    },
}


def _build_group_maps(constraints: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, str]]:
    groups = (
        constraints.get("schema", {})
        .get("groups", {})
        if isinstance(constraints, dict)
        else {}
    )
    element_to_group: dict[str, str] = {}
    normalized_groups: dict[str, list[str]] = {}
    for group_name, elems in groups.items():
        elems_list = [str(e) for e in (elems or [])]
        normalized_groups[str(group_name)] = elems_list
        for e in elems_list:
            element_to_group[e] = str(group_name)
    return normalized_groups, element_to_group


def _st_bounds(constraints: dict[str, Any], group_name: str) -> tuple[int, int]:
    st = constraints.get("stoichiometry", {}).get(group_name, {}) if isinstance(constraints, dict) else {}
    mn = int(st.get("min", 1))
    mx = int(st.get("max", 10))
    if mn > mx:
        mn, mx = mx, mn
    return mn, mx


def is_valid_hard_constrained_composition(
    composition: dict[str, int],
    constraints: dict[str, Any] | None = None,
    max_atoms: int | None = None,
) -> bool:
    if not composition:
        return False
    constraints = constraints or DEFAULT_HARD_CONSTRAINTS
    groups, element_to_group = _build_group_maps(constraints)
    required_groups = list(groups.keys())
    if not required_groups:
        return False

    group_counts: dict[str, int] = {g: 0 for g in required_groups}
    for el, amt in composition.items():
        if int(amt) < 1:
            return False
        g = element_to_group.get(str(el))
        if not g:
            return False
        group_counts[g] += int(amt)

    for g in required_groups:
        mn, mx = _st_bounds(constraints, g)
        if not (mn <= group_counts[g] <= mx):
            return False
        # Exactly one element chosen from each group.
        present_in_group = [el for el in composition.keys() if element_to_group.get(str(el)) == g]
        if len(present_in_group) != 1:
            return False

    total_atoms = sum(int(v) for v in composition.values())
    if max_atoms is not None and total_atoms > int(max_atoms):
        return False
    return True


def _sample_valid_hard_constrained_composition(
    constraints: dict[str, Any] | None = None,
    max_atoms: int | None = None,
    max_tries: int = 2000,
) -> dict[str, int]:
    constraints = constraints or DEFAULT_HARD_CONSTRAINTS
    groups, _ = _build_group_maps(constraints)
    group_names = list(groups.keys())
    if not group_names:
        raise ValueError("No groups found in hard constraints.")

    for _ in range(max_tries):
        comp: dict[str, int] = {}
        for g in group_names:
            elems = groups.get(g, [])
            if not elems:
                break
            el = random.choice(elems)
            mn, mx = _st_bounds(constraints, g)
            comp[el] = random.randint(mn, mx)
        if len(comp) != len(group_names):
            continue
        if is_valid_hard_constrained_composition(comp, constraints, max_atoms=max_atoms):
            return comp
    raise RuntimeError("Failed to sample a valid hard-constrained composition.")


def setup_logger(log_dir):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'monte_carlo_sampling_{timestamp}.log')
    
    logger = logging.getLogger('monte_carlo_sampler')
    logger.setLevel(logging.INFO)
    
    # 文件handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file


def generate_compound(n_elements, max_atoms):
    """
    生成一个随机化合物

    Args:
        n_elements: 元素数量 (2 或 3)
        max_atoms: 最大总原子数

    Returns:
        formula: 化学式字符串
        composition: {元素: 原子数} 字典
    """
    # 随机选择元素
    selected_elements = random.sample(ELEMENTS, n_elements)

    # 随机确定总原子数 (最小为n_elements，每个元素至少1个)
    # 小于等于max_atoms，所以上限是max_atoms
    min_atoms = n_elements
    total_atoms = random.randint(min_atoms, max_atoms)

    # 为每个元素分配原子数 (至少1个)
    remaining = total_atoms - n_elements
    atom_counts = [1] * n_elements

    # 随机分配剩余原子数
    for i in range(remaining):
        idx = random.randint(0, n_elements - 1)
        atom_counts[idx] += 1

    # 构建化学式
    composition = {}
    formula_parts = []
    for elem, count in zip(selected_elements, atom_counts):
        composition[elem] = count
        if count == 1:
            formula_parts.append(elem)
        else:
            formula_parts.append(f"{elem}{count}")

    formula = ''.join(formula_parts)

    return formula, composition, total_atoms


def composition_to_formula(composition):
    """Build a stable formula string from a composition dict."""
    parts = []
    for elem in sorted(composition.keys()):
        count = int(composition[elem])
        if count == 1:
            parts.append(elem)
        else:
            parts.append(f"{elem}{count}")
    return ''.join(parts)


def propose_neighbor(composition, max_atoms, constraints: dict[str, Any] | None = None):
    """Propose a neighboring composition for MCMC."""
    constraints = constraints or DEFAULT_HARD_CONSTRAINTS
    comp = composition.copy()
    groups, element_to_group = _build_group_maps(constraints)
    group_names = list(groups.keys())
    elements = list(comp.keys())
    if len(elements) != len(group_names):
        # Recover from invalid state.
        return _sample_valid_hard_constrained_composition(constraints, max_atoms=max_atoms)

    move = random.random()

    # 1) Substitute element inside the same group.
    if move < 0.4:
        g = random.choice(group_names)
        current_el = next((el for el in comp.keys() if element_to_group.get(el) == g), None)
        if not current_el:
            return composition
        candidates = [e for e in groups.get(g, []) if e != current_el]
        if not candidates:
            return composition
        new_el = random.choice(candidates)
        amt = comp.pop(current_el)
        comp[new_el] = amt
        return comp if is_valid_hard_constrained_composition(comp, constraints, max_atoms=max_atoms) else composition

    # 2) Adjust one group stoichiometry by +/-1 within bounds.
    if move < 1.0:
        g = random.choice(group_names)
        el = next((x for x in comp.keys() if element_to_group.get(x) == g), None)
        if not el:
            return composition
        delta = random.choice([-1, 1])
        new_val = comp[el] + delta
        mn, mx = _st_bounds(constraints, g)
        if not (mn <= new_val <= mx):
            return composition
        total_atoms = sum(comp.values()) - comp[el] + new_val
        if total_atoms > max_atoms:
            return composition
        comp[el] = new_val
        return comp if is_valid_hard_constrained_composition(comp, constraints, max_atoms=max_atoms) else composition

    return composition


def mcmc_sampling(
    n_samples,
    max_atoms,
    logger,
    score_fn,
    start_composition=None,
    burn_in=200,
    thin=5,
    max_steps=None,
    temperature=1.0,
    hard_constraints: dict[str, Any] | None = None,
):
    """
    MCMC sampling with Metropolis-Hastings.

    score_fn should return (log_weight, meta_dict).
    """
    if max_steps is None:
        max_steps = burn_in + n_samples * thin * 10

    hard_constraints = hard_constraints or DEFAULT_HARD_CONSTRAINTS
    if start_composition is None:
        start_composition = _sample_valid_hard_constrained_composition(
            hard_constraints, max_atoms=max_atoms
        )
    elif not is_valid_hard_constrained_composition(
        start_composition, hard_constraints, max_atoms=max_atoms
    ):
        start_composition = _sample_valid_hard_constrained_composition(
            hard_constraints, max_atoms=max_atoms
        )

    current = start_composition
    current_logw, current_meta = score_fn(current)

    samples = []
    generated_formulas = set()
    accepted = 0

    steps = 0
    while len(samples) < n_samples and steps < max_steps:
        steps += 1
        proposal = propose_neighbor(current, max_atoms, constraints=hard_constraints)
        if not is_valid_hard_constrained_composition(
            proposal, hard_constraints, max_atoms=max_atoms
        ):
            # Hard reject invalid proposal.
            continue
        proposal_logw, proposal_meta = score_fn(proposal)

        log_accept = (proposal_logw - current_logw) / max(temperature, 1e-9)
        if log_accept >= 0 or math.log(random.random()) < log_accept:
            current = proposal
            current_logw = proposal_logw
            current_meta = proposal_meta
            accepted += 1

        if steps <= burn_in:
            continue
        if (steps - burn_in) % thin != 0:
            continue

        formula = composition_to_formula(current)
        if formula in generated_formulas:
            continue
        generated_formulas.add(formula)

        n_elements = len(current)
        total_atoms = int(sum(current.values()))
        samples.append({
            'id': len(samples) + 1,
            'formula': formula,
            'composition': current.copy(),
            'n_elements': n_elements,
            'total_atoms': total_atoms,
            'log_weight': float(current_logw),
            **(current_meta or {})
        })

    if steps > 0:
        logger.info(f"MCMC acceptance rate: {accepted / steps:.3f}")
    logger.info(f"MCMC sampling complete: {len(samples)} samples")

    return samples


def monte_carlo_sampling(n_samples, max_atoms, logger, hard_constraints: dict[str, Any] | None = None):
    """
    蒙特卡洛随机采样
    
    Args:
        n_samples: 采样数量
        max_atoms: 最大原子数
        logger: 日志记录器
    
    Returns:
        samples: 采样结果列表
    """
    samples = []
    generated_formulas = set()  # 避免重复
    
    logger.info(f"开始蒙特卡洛采样...")
    logger.info(f"元素空间: {ELEMENTS}")
    logger.info(f"化合物类型: 二元、三元")
    logger.info(f"最大原子数: {max_atoms}")
    logger.info(f"目标采样数: {n_samples}")
    logger.info("-" * 60)
    
    attempts = 0
    max_attempts = n_samples * 10  # 防止死循环
    
    constraints = hard_constraints or DEFAULT_HARD_CONSTRAINTS
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1

        composition = _sample_valid_hard_constrained_composition(
            constraints, max_atoms=max_atoms
        )
        formula = composition_to_formula(composition)
        total_atoms = int(sum(composition.values()))
        n_elements = len(composition)
        
        # 检查是否重复
        if formula not in generated_formulas:
            generated_formulas.add(formula)
            samples.append({
                'id': len(samples) + 1,
                'formula': formula,
                'composition': composition,
                'n_elements': n_elements,
                'total_atoms': total_atoms
            })
            
            logger.info(f"[{len(samples):3d}] {formula:20s} | {n_elements}元 | 原子数: {total_atoms:2d} | {composition}")
    
    logger.info("-" * 60)
    logger.info(f"采样完成! 共生成 {len(samples)} 个化合物")
    
    # 统计
    binary = sum(1 for s in samples if s['n_elements'] == 2)
    ternary = sum(1 for s in samples if s['n_elements'] == 3)
    logger.info(f"二元化合物: {binary} 个")
    logger.info(f"三元化合物: {ternary} 个")
    
    return samples


def main():
    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    log_dir = os.path.join(project_root, 'logs')
    
    # 设置日志
    logger, log_file = setup_logger(log_dir)
    
    logger.info("=" * 60)
    logger.info("蒙特卡洛采样器 - 热电材料化合物生成")
    logger.info("=" * 60)
    
    # 执行采样 (不设置固定种子，每次结果不同)
    samples = monte_carlo_sampling(N_SAMPLES, MAX_ATOMS, logger)
    
    logger.info("=" * 60)
    logger.info(f"日志已保存: {log_file}")
    
    return samples


if __name__ == '__main__':
    main()

"""
理论文档更新器 - 基于成功案例更新理论原理文档

使用 AI 分析成功材料案例，提取关键特征和规律，
生成理论文档的更新版本，为下一轮迭代提供优化的理论指导。
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
THEORY_DOC_NAME = "Theoretical_principle_document.md"


class TheoryDocumentUpdater:
    """理论文档更新器"""
    
    def __init__(self, ai_client=None):
        """
        初始化文档更新器
        
        Args:
            ai_client: AI 客户端实例（如果为 None，会自动创建）
        """

    def __init__(self, ai_client=None):
        """
        初始化文档更新器
        
        Args:
            ai_client: AI 客户端实例（如果为 None，会自动创建）
        """
        if ai_client is None:
            try:
                # 尝试相对导入 (当作为包导入时)
                from .ai_client import AIClient
            except ImportError:
                try:
                    # 尝试绝对导入 (当作为脚本运行时，或者 sys.path 包含了 src)
                    from agents.ai_client import AIClient
                except ImportError:
                     # 最后的手段：添加到 sys.path
                    import sys
                    import os
                    # 假设文件在 src/agents/update_document.py
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    # 添加 src/agents
                    if current_dir not in sys.path:
                        sys.path.append(current_dir)
                    try:
                        from ai_client import AIClient
                    except ImportError:
                         # 尝试添加 src (父目录)
                         parent_dir = os.path.dirname(current_dir)
                         if parent_dir not in sys.path:
                             sys.path.append(parent_dir)
                         try:
                             from agents.ai_client import AIClient
                         except ImportError:
                             # 如果还是失败，可能 ai_client.py 就在同一目录下但导入名不同
                             # 假设它在同一目录
                             from ai_client import AIClient

            self.ai_client = AIClient()
        else:
            self.ai_client = ai_client
    
    def update_theory_document(
        self,
        original_doc_path: str,
        success_csv: str,
        output_dir: str = "llm/doc/v0.0.2",
        iteration_num: int = 1,
        results_root: str = "llm/results"
    ) -> Optional[str]:
        """
        基于成功案例更新理论文档
        
        Args:
            original_doc_path: 原始理论文档路径（如 doc/v0.0.1/Theoretical_principle_document.md）
            success_csv: 成功材料 CSV 文件路径
            output_dir: 输出目录
            iteration_num: 迭代轮次
            results_root: 结果存储根目录
            
        Returns:
            更新后的文档路径，如果没有数据则返回 None
        """
        logger.info("=" * 80)
        logger.info("理论文档更新 - 基于成功案例")
        logger.info("=" * 80)
        logger.info(f"原始文档: {original_doc_path}")
        logger.info(f"成功案例: {success_csv}")
        
        # 1. 读取原始理论文档（路径缺失时自动兜底）
        resolved_original_doc_path = self._resolve_original_doc_path(
            original_doc_path=original_doc_path,
            iteration_num=iteration_num,
            results_root=results_root,
        )
        if resolved_original_doc_path is None:
            logger.error(f"原始文档不存在: {original_doc_path}")
            return None
        if Path(original_doc_path).as_posix() != resolved_original_doc_path.as_posix():
            logger.warning(
                f"原始文档路径不存在，已自动回退: requested={original_doc_path}, resolved={resolved_original_doc_path}"
            )
        resolved_original_doc_path = self._ensure_canonical_doc_name(resolved_original_doc_path)

        with open(resolved_original_doc_path, 'r', encoding='utf-8') as f:
            original_doc = f.read()
        
        logger.info(f"✅ 原始文档已加载，长度: {len(original_doc)} 字符")
        
        # 2. 读取成功材料 CSV
        if not Path(success_csv).exists():
            logger.warning(f"CSV 文件不存在: {success_csv}")
            
            # 保存一个输出文档，说明文件不存在
            missing_output = "# 理论文档更新结果\n\n"
            missing_output += "**状态**: 跳过\n\n"
            missing_output += f"**原因**: 成功材料文件不存在\n\n"
            missing_output += f"**预期文件**: {success_csv}\n\n"
            missing_output += "**建议**: 请确保已完成成功材料提取步骤\n\n"
            
            output_file = self._save_output(missing_output, iteration_num, results_root)
            logger.info(f"💾 LLM 输出已保存（文件缺失）: {output_file}")
            
            # 复制原始文档到输出位置
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            doc_file = output_path / THEORY_DOC_NAME
            
            with open(resolved_original_doc_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            if resolved_original_doc_path.resolve() != doc_file.resolve():
                with open(doc_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                logger.info(f"📄 理论文档已保持不变: {doc_file}")
            else:
                logger.info(f"📄 理论文档保持不变（已在输出位置）: {doc_file}")
            
            return str(doc_file)
        
        try:
            df = pd.read_csv(success_csv, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"读取 CSV 失败: {e}")
            
            # 保存错误信息
            read_error_output = "# 理论文档更新结果\n\n"
            read_error_output += "**状态**: 失败\n\n"
            read_error_output += f"**错误**: 无法读取CSV文件\n\n"
            read_error_output += f"**详细信息**: {str(e)}\n\n"
            read_error_output += f"**文件路径**: {success_csv}\n\n"
            
            output_file = self._save_output(read_error_output, iteration_num, results_root)
            logger.info(f"💾 LLM 输出已保存（读取错误）: {output_file}")
            
            # 复制原始文档到输出位置
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            doc_file = output_path / THEORY_DOC_NAME
            
            with open(resolved_original_doc_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            if resolved_original_doc_path.resolve() != doc_file.resolve():
                with open(doc_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                logger.info(f"📄 理论文档已保持不变: {doc_file}")
            else:
                logger.info(f"📄 理论文档保持不变（已在输出位置）: {doc_file}")
            
            return str(doc_file)
        
        if len(df) == 0:
            logger.warning("CSV 文件为空，没有成功材料可供学习")
            
            # 保存说明文档
            empty_output = "# 理论文档更新结果\n\n"
            empty_output += "**状态**: 跳过\n\n"
            empty_output += "**原因**: 本轮次未找到成功材料，理论文档保持不变\n\n"
            empty_output += f"**原始文档**: {resolved_original_doc_path}\n\n"
            
            output_file = self._save_output(empty_output, iteration_num, results_root)
            logger.info(f"💾 LLM 输出已保存: {output_file}")
            
            # 复制原始文档到输出位置，保持文档的连续性
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            doc_file = output_path / THEORY_DOC_NAME
            
            # 读取原始文档
            with open(resolved_original_doc_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 保存到输出位置（如果路径相同则不需要复制）
            if resolved_original_doc_path.resolve() != doc_file.resolve():
                with open(doc_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                logger.info(f"📄 理论文档已保持不变: {doc_file}")
            else:
                logger.info(f"📄 理论文档保持不变（已在输出位置）: {doc_file}")
            
            # 返回文档路径而非None
            return str(doc_file)
        
        logger.info(f"✅ 找到 {len(df)} 个成功材料")
        
        # 3. 格式化成功案例信息
        success_info = self._format_success_materials(df)
        
        # 4. 构建文档更新 prompt
        prompt = self._build_update_prompt(original_doc, success_info)
        
        # 5. 保存 LLM 输入
        input_file = self._save_input(prompt, iteration_num, results_root)
        logger.info(f"💾 LLM 输入已保存: {input_file}")
        
        # 6. 调用 AI 生成更新后的文档
        logger.info("🔍 AI 正在分析成功案例并更新理论文档...")
        
        try:
            updated_doc = self.ai_client.chat(
                prompt=prompt,
                model_id=self.ai_client.get_default_model("theory_update"),
                temperature=self.ai_client.get_default_temperature("theory_update"),
                max_tokens=8000
            )
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            # 保存错误信息到输出文档
            error_output = "# 理论文档更新结果\n\n"
            error_output += "**状态**: 失败\n\n"
            error_output += f"**错误**: {str(e)}\n\n"
            error_output += f"**成功材料数量**: {len(df)}\n\n"
            error_output += "**建议**: 请检查 API 配置或稍后重试\n\n"
            
            output_file = self._save_output(error_output, iteration_num, results_root)
            logger.info(f"💾 LLM 输出已保存（错误信息）: {output_file}")
            
            # 返回 None 表示更新失败，但输出文档已保存
            return None
        
        # 7. 保存 LLM 输出
        output_file = self._save_output(updated_doc, iteration_num, results_root)
        logger.info(f"💾 LLM 输出已保存: {output_file}")
        
        # 8. 保存更新后的文档（使用固定文件名，不带时间戳和版本号）
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        doc_file = output_path / THEORY_DOC_NAME
        
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(updated_doc)
        
        logger.info(f"✅ 更新后的文档已保存: {doc_file}")
        logger.info("=" * 80)
        
        return str(doc_file)

    def _resolve_original_doc_path(
        self,
        original_doc_path: str,
        iteration_num: int,
        results_root: str,
    ) -> Optional[Path]:
        requested = Path(original_doc_path)
        if requested.exists():
            return requested

        parent = requested.parent
        if parent.exists():
            md_candidates = [p for p in parent.glob("*.md") if p.is_file()]
            exact_name = [p for p in md_candidates if p.name == THEORY_DOC_NAME]
            if exact_name:
                return exact_name[0]
            if len(md_candidates) == 1:
                return md_candidates[0]

        # 兜底：回溯最近一轮文档（results 优先，其次 doc 版本目录）
        recovered = self._find_latest_available_doc(iteration_num=iteration_num, results_root=results_root)
        if recovered is None:
            return None

        try:
            parent.mkdir(parents=True, exist_ok=True)
            if not requested.exists():
                requested.write_text(recovered.read_text(encoding="utf-8"), encoding="utf-8")
                logger.warning(f"路径缺失，已自动复制最近文档到目标位置: {requested} <- {recovered}")
                return requested
        except Exception as exc:
            try:
                fallback_target = parent / THEORY_DOC_NAME
                parent.mkdir(parents=True, exist_ok=True)
                fallback_target.write_text(recovered.read_text(encoding="utf-8"), encoding="utf-8")
                logger.warning(
                    f"目标路径写入失败，已改为目录标准文件名: {fallback_target} <- {recovered}, error={exc}"
                )
                return fallback_target
            except Exception as fallback_exc:
                logger.warning(
                    f"自动复制最近文档失败，改为直接使用回溯文档: {recovered}, "
                    f"error={exc}, fallback_error={fallback_exc}"
                )
        return recovered

    def _find_latest_available_doc(self, iteration_num: int, results_root: str) -> Optional[Path]:
        cwd = Path.cwd()
        results_root_path = Path(results_root)
        if not results_root_path.is_absolute():
            results_root_path = cwd / results_root_path
        doc_root_path = results_root_path.parent / "doc"

        max_back = max(iteration_num, 1)
        for i in range(max_back - 1, -1, -1):
            version_doc = doc_root_path / f"v0.0.{i}" / THEORY_DOC_NAME
            if version_doc.exists():
                return version_doc
            if i == 0:
                # v0.0.0 下如果文件名乱码但只有一个 md，也接受
                v0 = doc_root_path / "v0.0.0"
                if v0.exists():
                    md_candidates = [p for p in v0.glob("*.md") if p.is_file()]
                    if len(md_candidates) == 1:
                        return md_candidates[0]
        return None

    def _ensure_canonical_doc_name(self, source_path: Path) -> Path:
        if source_path.name == THEORY_DOC_NAME:
            return source_path

        canonical = source_path.parent / THEORY_DOC_NAME
        if canonical.exists():
            return canonical

        try:
            canonical.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.warning(f"检测到非标准文档文件名，已自动生成标准副本: {canonical}")
            return canonical
        except Exception as exc:
            logger.warning(f"自动生成标准文件名副本失败，继续使用原文件: {source_path}, error={exc}")
            return source_path
    
    def _format_success_materials(self, df: pd.DataFrame) -> str:
        """Format successful materials information - using only required fields"""
        lines = []
        lines.append("# Successful Cases List")
        lines.append("")
        lines.append(f"**Number of Cases**: {len(df)}")
        lines.append("")

        required_fields = ["composition", "thermal_conductivity_w_mk", "structure"]

        for idx, row in df.iterrows():
            formula = row.get("composition", row.get("formula", "Unknown"))
            lines.append(f"## Case {idx + 1}: {formula}")
            lines.append("")

            kappa = row.get(
                "thermal_conductivity_w_mk",
                row.get("thermal_conductivity", row.get("热导率(W/m·K)", row.get("热导率 (W/m·K)")))
            )
            if pd.notna(kappa):
                lines.append(f"- **Thermal Conductivity**: {kappa} W/(m·K)")
            else:
                lines.append("- **Thermal Conductivity**: Data missing")

            structure_data = row.get("structure", row.get("Structure"))
            if pd.notna(structure_data):
                lines.append("- **Crystal Structure (Pymatgen)**:")
                lines.append("```")
                lines.append(f"{structure_data}")
                lines.append("```")
            else:
                lines.append("- **Crystal Structure**: Data missing or not converted to Pymatgen format")

            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("## Data Completeness")
        lines.append("")

        aliases = {
            "composition": ["composition", "formula"],
            "thermal_conductivity_w_mk": ["thermal_conductivity_w_mk", "thermal_conductivity", "热导率(W/m·K)", "热导率 (W/m·K)"],
            "structure": ["structure", "Structure"],
        }
        for field in required_fields:
            matched = next((c for c in aliases[field] if c in df.columns), None)
            if matched:
                valid_count = df[matched].notna().sum()
                lines.append(f"- **{field}**: {valid_count}/{len(df)} cases have data")
            else:
                lines.append(f"- **{field}**: Field missing")

        return "\n".join(lines)
    
    def _build_update_prompt(self, original_doc: str, success_info: str) -> str:
        """Build document update prompt"""
        prompt = f"""# Theoretical Document Update Task

## Original Theoretical Document

{original_doc}

---

## Successful Material Cases

{success_info}

---

## Update Task

Based on successful cases, update the theoretical principles document. Use **theoretical summary format**, not specific case lists.

**Update Focus**:

1. **Analyze Common Patterns**:
   - Element combination patterns
   - Mass contrast distribution
   - Lone pair electron effects
   - Structural feature commonalities

2. **Update Section 4**:
   - 4.2 Theoretical Summary of Successful Cases: Extract patterns (e.g., "Te-containing ternary compounds show high success rate")
   - 4.3 Theoretical Optimization Records: Document theoretical adjustments (e.g., "Increased Te element weight")

3. **Format Requirements**:
   - ❌ Avoid: Specific case tables, version numbers, timestamps, document metadata
   - ✅ Use: Pure theoretical summary statements
   - ❌ Do not add: "Version", "Update date", "Document version control", etc.
   - ✅ Only add brief annotations like `[Updated this round]` at specific update locations

4. **Output Requirements**:
   - Output complete Markdown document directly
   - Maintain original document core structure (Sections 1-6)
   - Only update Section 4 content
   - Do not add any metadata, version notes, or update logs
   - Do not add extra explanations or descriptions

**Output**: Pure theoretical document content without any metadata.
"""
        return prompt
    
    def _save_input(self, prompt: str, iteration_num: int, results_root: str = "results") -> str:
        """保存 LLM 输入到文件"""
        output_dir = Path(f'{results_root}/iteration_{iteration_num}/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用配置文件中定义的文件名
        input_file = output_dir / 'llm_theory_update_input.md'
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        return str(input_file)
    
    def _save_output(self, response: str, iteration_num: int, results_root: str = "results") -> str:
        """保存 LLM 输出到文件"""
        output_dir = Path(f'{results_root}/iteration_{iteration_num}/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用配置文件中定义的文件名
        output_file = output_dir / 'llm_theory_update_output.md'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)
        
        return str(output_file)


def update_theory_from_success(
    original_doc_path: str,
    success_csv: str,
    output_dir: str = "llm/doc/v0.0.2",
    iteration_num: int = 1,
    results_root: str = "llm/results"
) -> Optional[str]:
    """
    从成功材料更新理论文档的便捷函数
    
    Args:
        original_doc_path: 原始理论文档路径
        success_csv: 成功材料 CSV 文件路径
        output_dir: 输出目录
        iteration_num: 迭代轮次
        results_root: 结果存储根目录
        
    Returns:
        更新后的文档路径
    """
    updater = TheoryDocumentUpdater()
    return updater.update_theory_document(
        original_doc_path, 
        success_csv, 
        output_dir,
        iteration_num,
        results_root
    )


if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    result = update_theory_from_success(
        original_doc_path='llm/doc/v0.0.0/Theoretical_principle_document.md',
        success_csv='llm/results/iteration_1/success_examples/stable_materials_deduped.csv',
        output_dir='llm/doc/v0.0.1',
        iteration_num=1
    )
    
    if result:
        print(f"\n✅ 文档更新完成: {result}")
    else:
        print("\n❌ 文档更新失败")

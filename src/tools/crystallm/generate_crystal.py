import subprocess
import os
import sys
import time
import json
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import structlog
# from CrystaLLM.bin import make_prompt_file, sample, postprocess

logger = structlog.get_logger(__name__)

class CrystalStructureGenerator:
    def __init__(self, composition, params=None, progress_callback: Optional[Callable[[str, float], None]] = None, spacegroup: Optional[str] = None):
        self.composition = composition
        self.progress_callback = progress_callback
        self.spacegroup = spacegroup  # 空间群约束（可选）

        # Get the directory where this file is located
        self._module_dir = Path(__file__).parent.absolute()

        # 默认参数
        default_params = {
            "bin_dir": os.path.join(os.path.abspath('.'), 'CrystaLLM', 'bin'),
            'prompt_dir': 'Batch_Prompts',
            'num_samples': 20,
            'top_k': 20,
            'max_new_tokens': 3000,
            'device': 'cuda',
            # 生成参数
            'out_dir': 'pre-trained-model/crystallm_v1_small',
            'target': 'file',
            'generate_dir': None,  # 默认为None,会根据composition自动生成
            # 后处理参数
            'postprocess_input_dir': None,  # 默认为None,会使用generate_dir
            'postprocess_output_dir': None  # 默认为None,会根据composition自动生成
        }
        
        # 更新参数
        self.params = default_params
        if params:
            self.params.update(params)
            
        # 设置提示词参数
        self.prompt_params = {
            'prompt_dir': self.params['prompt_dir'],
            'prompt_name': f'{composition}.txt'
        }
        
        # 设置生成参数
        # 设置生成参数
        use_exact = self.params.get('exact_paths', False)
        
        self.generated_params = {
            'out_dir': self.params['out_dir'],
            'num_samples': self.params['num_samples'],
            'top_k': self.params['top_k'],
            'max_new_tokens': self.params['max_new_tokens'],
            'device': self.params['device'],
            'target': self.params['target'],
            'generate_dir': self.params['generate_dir'] if use_exact else os.path.join(self.params['generate_dir'], composition)
        }
        
        # 设置后处理参数
        self.postprocess_params = {
            'input_dir': self.params['postprocess_input_dir'] if use_exact else os.path.join(self.params['postprocess_input_dir'], composition),
            'output_dir': self.params['postprocess_output_dir'] if use_exact else os.path.join(self.params['postprocess_output_dir'], composition)
        }
        
    def _get_subprocess_env(self):
        """Get environment variables for subprocess with PYTHONPATH set."""
        env = os.environ.copy()
        # Add module directory to PYTHONPATH so crystallm module can be imported
        pythonpath = str(self._module_dir)
        if 'PYTHONPATH' in env:
            pythonpath = pythonpath + os.pathsep + env['PYTHONPATH']
        env['PYTHONPATH'] = pythonpath
        return env

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is provided."""
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def _structure_to_frontend_format(self, structure: Structure, structure_id: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert pymatgen Structure to frontend-compatible format."""
        try:
            # 获取原胞信息
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
            primitive_structure = analyzer.get_primitive_standard_structure()
            spacegroup = analyzer.get_space_group_symbol()
            space_group_no = analyzer.get_space_group_number()

            # 获取晶格参数
            lattice = primitive_structure.lattice

            # 转换原子信息
            atoms = []
            for site in primitive_structure.sites:
                atoms.append({
                    "element": str(site.specie),
                    "position": [float(site.coords[0]), float(site.coords[1]), float(site.coords[2])]
                })

            # 生成CIF内容
            cif_content = ""
            try:
                cif_writer = CifWriter(primitive_structure, symprec=0.1)
                cif_content = str(cif_writer)
            except Exception as e:
                print(f"Warning: Could not generate CIF for {structure_id}: {e}")

            # 获取惯胞数据用于切换
            conventional_structure = analyzer.get_conventional_standard_structure()
            conv_lattice = conventional_structure.lattice
            conventional_data = {
                "latticeParameters": {
                    "a": float(conv_lattice.a),
                    "b": float(conv_lattice.b),
                    "c": float(conv_lattice.c),
                    "alpha": float(conv_lattice.alpha),
                    "beta": float(conv_lattice.beta),
                    "gamma": float(conv_lattice.gamma)
                },
                "atoms": [],
                "volume": float(conv_lattice.volume),
                "numAtoms": len(conventional_structure)
            }
            for site in conventional_structure.sites:
                conventional_data["atoms"].append({
                    "element": str(site.specie),
                    "position": [float(x) for x in site.frac_coords.tolist()]
                })

            # 获取原胞数据用于切换
            primitive_data = {
                "latticeParameters": {
                    "a": float(lattice.a),
                    "b": float(lattice.b),
                    "c": float(lattice.c),
                    "alpha": float(lattice.alpha),
                    "beta": float(lattice.beta),
                    "gamma": float(lattice.gamma)
                },
                "atoms": [],
                "volume": float(lattice.volume),
                "numAtoms": len(atoms)
            }
            for site in primitive_structure.sites:
                primitive_data["atoms"].append({
                    "element": str(site.specie),
                    "position": [float(x) for x in site.frac_coords.tolist()]
                })

            # 构建前端格式的结构数据
            frontend_structure = {
                "id": structure_id,
                "formula": primitive_structure.composition.reduced_formula,
                "spaceGroup": spacegroup,
                "latticeParameters": {
                    "a": float(lattice.a),
                    "b": float(lattice.b),
                    "c": float(lattice.c),
                    "alpha": float(lattice.alpha),
                    "beta": float(lattice.beta),
                    "gamma": float(lattice.gamma)
                },
                "atoms": atoms,
                "properties": {
                    "density": float(primitive_structure.density),
                    "volume": float(primitive_structure.volume),
                    "numAtoms": len(atoms),
                    "atomCount": len(atoms),  # Keep for backward compatibility
                    "spaceGroupNumber": space_group_no,
                    "crystalSystem": analyzer.get_crystal_system() if analyzer else None
                },
                "source": {
                    "database": "Generated",
                    "generator": "CrystaLLM",
                    "composition": self.composition
                },
                "metadata": {
                    "spaceGroupNumber": space_group_no,  # Keep for backward compatibility
                    "timestamp": datetime.now().isoformat(),
                    "generationParams": self.params.copy(),
                    **(metadata or {})
                },
                "cifContent": cif_content,  # Unified field name
                "cellTypes": {
                    "primitive": primitive_data,
                    "conventional": conventional_data
                },
                "currentCellType": "primitive"  # Default to primitive
            }

            return frontend_structure
            
        except Exception as e:
            print(f"Error converting structure {structure_id}: {e}")
            return None

    def generate_prompt(self):
        self._report_progress("生成提示词文件...", 0.1)
        os.makedirs(self.prompt_params['prompt_dir'], exist_ok=True)
        prompt_path = os.path.join(self.prompt_params['prompt_dir'], self.prompt_params['prompt_name'])
        env = self._get_subprocess_env()
        # Use current Python interpreter
        python_exe = sys.executable

        # 构建命令参数
        cmd_args = [python_exe, os.path.join(self.params['bin_dir'], 'make_prompt_file.py'), self.composition, prompt_path]

        # 如果指定了空间群，添加 --spacegroup 参数
        if self.spacegroup:
            cmd_args.extend(['--spacegroup', self.spacegroup])
            logger.info(f"🔬 使用空间群约束: {self.spacegroup}")

        logger.info(f"🚀 开始生成提示词: {self.composition}")
        logger.info(f"命令: {' '.join(cmd_args)}")
        print(f"\n[DEBUG] 正在执行提示词生成: {self.composition}", flush=True)
        print(f"[DEBUG] 命令: {' '.join(cmd_args)}", flush=True)
        print(f"[DEBUG] 工作目录: {os.getcwd()}", flush=True)
        print(f"[DEBUG] bin目录: {self.params['bin_dir']}", flush=True)
        sys.stdout.flush()
        
        try:
            result = subprocess.run(
                cmd_args, 
                env=env, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            # 打印输出用于调试
            if result.stdout:
                print(f"[STDOUT] 提示词生成输出:", flush=True)
                print(result.stdout, flush=True)
            if result.stderr:
                print(f"[STDERR] 提示词生成错误:", flush=True)
                print(result.stderr, flush=True)
                
            if result.returncode != 0:
                logger.error(f"❌ 提示词生成失败 (返回码: {result.returncode})")
                logger.error(f"错误信息: {result.stderr}")
                raise RuntimeError(f"提示词生成失败: {result.stderr}")
            
            print(f"[DEBUG] ✅ 提示词文件已生成: {prompt_path}", flush=True)
            logger.info(f"✅ 提示词文件已生成: {prompt_path}")
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 提示词生成超时 (>120s)")
            raise RuntimeError("提示词生成超时")
        except Exception as e:
            logger.error(f"❌ 提示词生成异常: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.generated_params['start'] = f"FILE:{prompt_path}"
        self._report_progress("提示词文件生成完成", 0.2)
        
    def generate_structures(self):
        self._report_progress("开始生成晶体结构...", 0.3)
        os.makedirs(self.generated_params['generate_dir'], exist_ok=True)
        start_time = time.time()
        # Use current Python interpreter
        python_exe = sys.executable
        cmd_args = [python_exe, os.path.join(self.params['bin_dir'], 'sample.py')] + [f"{k}={v}" for k,v in self.generated_params.items()]
        env = self._get_subprocess_env()
        
        logger.info(f"🔮 开始生成 {self.params['num_samples']} 个晶体结构...")
        logger.info(f"命令: {' '.join(cmd_args[:3])}... (参数已简化)")
        logger.info(f"设备: {self.params['device']}")
        print(f"\n[DEBUG] 正在执行结构生成...", flush=True)
        print(f"[DEBUG] 输出目录: {self.generated_params['generate_dir']}", flush=True)
        try:
            # 直接使用communicate并打印输出
            result = subprocess.run(cmd_args, env=env, capture_output=True, text=True, timeout=1800)  # 30分钟超时
            
            # 无论成功失败，都打印输出用于调试
            if result.stdout:
                print(f"\n[STDOUT] 结构生成输出:", flush=True)
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}", flush=True)
            
            if result.stderr:
                print(f"\n[STDERR] 结构生成错误输出:", flush=True)
                for line in result.stderr.split('\n'):
                    if line.strip():
                        print(f"  {line}", flush=True)
            
            if result.returncode != 0:
                logger.error(f"❌ 结构生成失败 (返回码: {result.returncode})")
                logger.error(f"错误信息: {result.stderr}")
                raise RuntimeError(f"结构生成失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 结构生成超时 (>30分钟)")
            raise RuntimeError("结构生成超时")
        
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"✅ 结构生成完成，耗时: {elapsed:.2f}秒 ({elapsed/60:.1f}分钟)")
        self._report_progress("晶体结构生成完成", 0.6)
        
    def postprocess_structures(self):
        self._report_progress("开始后处理...", 0.7)
        os.makedirs(self.postprocess_params['output_dir'], exist_ok=True)
        # Use current Python interpreter
        python_exe = sys.executable
        cmd_args = [python_exe, os.path.join(self.params['bin_dir'], 'postprocess.py'),
                   self.postprocess_params['input_dir'],
                   self.postprocess_params['output_dir']]
        env = self._get_subprocess_env()
        
        logger.info(f"⚙️ 开始后处理结构...")
        logger.info(f"命令: {' '.join(cmd_args)}")
        try:
            result = subprocess.run(cmd_args, env=env, capture_output=True, text=True, timeout=600)  # 10分钟超时
            if result.returncode != 0:
                logger.error(f"❌ 后处理失败: {result.stderr}")
                raise RuntimeError(f"后处理失败: {result.stderr}")
            if result.stdout:
                logger.debug(f"后处理输出: {result.stdout}")
            logger.info(f"✅ 后处理完成")
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 后处理超时 (>10分钟)")
            raise RuntimeError("后处理超时")
        self._report_progress("后处理完成", 0.8)
        
    def analyze_structures(self, directory):
        data = []
        for i in range(self.generated_params['num_samples']):
            cif_file = os.path.join(directory, f'sample_{i+1}.cif')
            if os.path.exists(cif_file):
                try:
                    with open(cif_file, 'r') as f:
                        cif_content = f.read()
                    structure = Structure.from_file(cif_file)
                    
                    # 获取原胞
                    analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
                    primitive_structure = analyzer.get_primitive_standard_structure()
                    
                    # 获取空间群信息
                    spacegroup = analyzer.get_space_group_symbol()
                    space_group_no = analyzer.get_space_group_number()
                    
                    # 获取晶胞参数
                    lattice = primitive_structure.lattice
                        
                    data.append({
                        '结构编号': i+1,
                        '化学式': primitive_structure.composition.reduced_formula,
                        '原子数': len(primitive_structure),
                        '晶胞参数a': f"{lattice.a:.3f}",
                        '晶胞参数b': f"{lattice.b:.3f}", 
                        '晶胞参数c': f"{lattice.c:.3f}",
                        '晶胞角度α': f"{lattice.alpha:.1f}°",
                        '晶胞角度β': f"{lattice.beta:.1f}°",
                        '晶胞角度γ': f"{lattice.gamma:.1f}°",
                        "空间群": spacegroup,
                        "空间群号": space_group_no,
                        "CIF文件内容": cif_content
                    })
                except ValueError as e:
                    print(f"\n警告: 结构文件 {cif_file} 解析失败: {str(e)}")
            else:
                print(f"\n警告: 结构文件 {cif_file} 不存在")
                
        if not data:
            print(f"\n警告: 目录 {directory} 中没有有效的结构文件")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        return df
    
    def generate_frontend_structures(self, directory: str) -> List[Dict[str, Any]]:
        """Generate frontend-compatible structure data from CIF files."""
        self._report_progress("转换结构数据格式...", 0.9)
        structures = []
        
        for i in range(self.generated_params['num_samples']):
            cif_file = os.path.join(directory, f'sample_{i+1}.cif')
            if os.path.exists(cif_file):
                try:
                    structure = Structure.from_file(cif_file)
                    structure_id = f"{self.composition}_generated_{i+1}_{int(datetime.now().timestamp())}"
                    
                    # 添加一些元数据
                    metadata = {
                        "sampleIndex": i + 1,
                        "cifFile": cif_file,
                        "generationMethod": "CrystaLLM_AI"
                    }
                    
                    frontend_structure = self._structure_to_frontend_format(structure, structure_id, metadata)
                    if frontend_structure:
                        structures.append(frontend_structure)
                        
                except Exception as e:
                    print(f"Warning: Could not process structure {i+1}: {e}")
        
        self._report_progress("结构转换完成", 1.0)
        return structures
    
    def export_structures_json(self, structures: List[Dict[str, Any]], output_file: str):
        """Export structures to JSON file."""
        output_data = {
            "composition": self.composition,
            "generationParams": self.params,
            "timestamp": datetime.now().isoformat(),
            "totalStructures": len(structures),
            "structures": structures
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(structures)} structures to {output_file}")
        
    def run_pipeline(self, export_json: bool = False, json_output_dir: str = None):
        """运行完整的生成流水线，支持前端数据格式导出"""
        try:
            self._report_progress("开始晶体生成流水线", 0.0)
            
            self.generate_prompt()
            self.generate_structures() 
            self.postprocess_structures()
            
            # 生成传统格式的分析结果
            raw_df = self.analyze_structures(self.generated_params['generate_dir'])
            processed_df = self.analyze_structures(self.postprocess_params['output_dir'])
            
            # 如果需要，生成前端兼容的结构数据
            frontend_structures = []
            if export_json:
                # 优先使用后处理的结构，如果没有则使用原始结构
                structure_dir = self.postprocess_params['output_dir'] if not processed_df.empty else self.generated_params['generate_dir']
                frontend_structures = self.generate_frontend_structures(structure_dir)
                
                # 导出JSON文件
                if json_output_dir and frontend_structures:
                    os.makedirs(json_output_dir, exist_ok=True)
                    json_filename = f"{self.composition}_structures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    json_path = os.path.join(json_output_dir, json_filename)
                    self.export_structures_json(frontend_structures, json_path)
            
            self._report_progress("流水线完成", 1.0)
            return {
                "raw_dataframe": raw_df,
                "processed_dataframe": processed_df,
                "frontend_structures": frontend_structures,
                "success": True,
                "message": f"成功生成 {len(frontend_structures)} 个晶体结构"
            }
            
        except Exception as e:
            error_msg = f"处理 {self.composition} 时发生错误: {str(e)}"
            print(f"\n错误: {error_msg}")
            self._report_progress(f"错误: {error_msg}", 0.0)
            return {
                "raw_dataframe": pd.DataFrame(),
                "processed_dataframe": pd.DataFrame(),
                "frontend_structures": [],
                "success": False,
                "message": error_msg
            }

def main():
    """示例主函数，展示如何使用增强的CrystalStructureGenerator"""
    # 进度回调函数
    def progress_callback(message: str, progress: float):
        print(f"[{progress:.1%}] {message}")
    
    # 自定义参数
    custom_params = {
        'prompt_dir': 'my_prompts',
        'num_samples': 10,  # 减少样本数量便于测试
        'top_k': 10,
        'max_new_tokens': 2000,
        'device': 'cuda',
        'out_dir': 'pre-trained-model/crystallm_v1_small',
        # 自定义生成目录
        'generate_dir': 'my_generated_structures',
        # 自定义后处理目录
        'postprocess_input_dir': 'my_generated_structures',
        'postprocess_output_dir': 'my_processed_structures'
    }
    
    # 创建生成器实例
    generator = CrystalStructureGenerator("C8", params=custom_params, progress_callback=progress_callback)
    
    # 运行完整流水线，包括前端格式导出
    result = generator.run_pipeline(export_json=True, json_output_dir='frontend_exports')
    
    if result['success']:
        print(f"\n✅ 生成完成!")
        print(f"📊 原始结构: {len(result['raw_dataframe'])} 个")
        print(f"🔧 后处理结构: {len(result['processed_dataframe'])} 个")
        print(f"🌐 前端格式结构: {len(result['frontend_structures'])} 个")
        
        # 显示前几个结构的信息
        if result['frontend_structures']:
            print(f"\n前3个生成的结构:")
            for i, structure in enumerate(result['frontend_structures'][:3]):
                print(f"  {i+1}. {structure['formula']} - {structure['spaceGroup']} - {structure['properties']['atomCount']} 原子")
    else:
        print(f"❌ 生成失败: {result['message']}")

# 提供一个便于外部调用的函数
def generate_structures_for_composition(composition: str, num_samples: int = 5, export_json: bool = True, progress_callback=None, spacegroup: Optional[str] = None) -> Dict[str, Any]:
    """
    为指定化学组成生成晶体结构的便捷函数

    Args:
        composition: 化学组成，如 "Li2O", "CaTiO3"
        num_samples: 生成的样本数量
        export_json: 是否导出前端兼容的JSON格式
        progress_callback: 进度回调函数
        spacegroup: 空间群约束（可选），如 "P4/nmm", "Fd-3m" 等

    Returns:
        包含生成结果的字典
    """
    custom_params = {
        'prompt_dir': f'prompts_{composition}',
        'num_samples': num_samples,
        'top_k': 10,
        'max_new_tokens': 2000,
        'device': 'cuda',
        'out_dir': 'pre-trained-model/crystallm_v1_small',
        'generate_dir': f'generated_structures_{composition}',
        'postprocess_input_dir': f'generated_structures_{composition}',
        'postprocess_output_dir': f'processed_structures_{composition}'
    }

    generator = CrystalStructureGenerator(composition, params=custom_params, progress_callback=progress_callback, spacegroup=spacegroup)
    return generator.run_pipeline(export_json=export_json, json_output_dir='frontend_exports')

def convert_cif_to_frontend_format(cif_content: str, filename: str, composition: str, generation_id: str) -> Dict[str, Any]:
    """
    将CIF内容转换为前端兼容的结构格式（优化版，移除冗余的 base64 编码）
    
    Args:
        cif_content: CIF文件内容
        filename: CIF文件名
        composition: 化学组成
        generation_id: 生成ID
        
    Returns:
        前端兼容的结构字典
    """
    try:
        import uuid
        from datetime import datetime
        
        # 生成唯一ID
        structure_id = str(uuid.uuid4())
        
        # 从CIF内容提取基本信息
        formula = composition  # 使用输入的化学式
        
        # 尝试从CIF中提取空间群信息
        space_group = "Unknown"
        for line in cif_content.split('\n'):
            if '_space_group_name_H-M_alt' in line or '_symmetry_space_group_name_H-M' in line:
                parts = line.split()
                if len(parts) > 1:
                    space_group = ' '.join(parts[1:]).strip('"\'')
                break
        
        # 使用 pymatgen 解析 CIF 以获取详细结构信息
        try:
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            struct = Structure.from_str(cif_content, fmt="cif")

            # 使用 SpacegroupAnalyzer 获取标准化结构
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5.0)
                primitive_structure = sga.get_primitive_standard_structure()
                conventional_structure = sga.get_conventional_standard_structure()
                space_group = sga.get_space_group_symbol()
                space_group_number = sga.get_space_group_number()
                crystal_system = sga.get_crystal_system()

                # 使用原胞作为显示结构
                display_struct = primitive_structure
                logger.info(f"✅ Analyzed structure: {composition}")
                logger.info(f"   Primitive: {len(primitive_structure)} sites")
                logger.info(f"   Conventional: {len(conventional_structure)} sites")

                # 准备原胞和惯胞的cellTypes数据（用于前端切换）
                prim_lattice = primitive_structure.lattice
                primitive_data = {
                    "latticeParameters": {
                        "a": round(prim_lattice.a, 6),
                        "b": round(prim_lattice.b, 6),
                        "c": round(prim_lattice.c, 6),
                        "alpha": round(prim_lattice.alpha, 6),
                        "beta": round(prim_lattice.beta, 6),
                        "gamma": round(prim_lattice.gamma, 6)
                    },
                    "atoms": [],
                    "volume": float(prim_lattice.volume),
                    "numAtoms": len(primitive_structure)
                }
                for site in primitive_structure:
                    primitive_data["atoms"].append({
                        "element": site.species_string,
                        "position": [round(x, 6) for x in site.frac_coords.tolist()],  # 分数坐标
                        "occupancy": 1.0
                    })

                conv_lattice = conventional_structure.lattice
                conventional_data = {
                    "latticeParameters": {
                        "a": round(conv_lattice.a, 6),
                        "b": round(conv_lattice.b, 6),
                        "c": round(conv_lattice.c, 6),
                        "alpha": round(conv_lattice.alpha, 6),
                        "beta": round(conv_lattice.beta, 6),
                        "gamma": round(conv_lattice.gamma, 6)
                    },
                    "atoms": [],
                    "volume": float(conv_lattice.volume),
                    "numAtoms": len(conventional_structure)
                }
                for site in conventional_structure:
                    conventional_data["atoms"].append({
                        "element": site.species_string,
                        "position": [round(x, 6) for x in site.frac_coords.tolist()],  # 分数坐标
                        "occupancy": 1.0
                    })
            except Exception as sga_error:
                logger.warning(f"⚠️ SpacegroupAnalyzer failed: {sga_error}, using original structure")
                display_struct = struct
                # Keep the space_group from CIF parsing above
                space_group_number = None
                crystal_system = "triclinic"
                primitive_data = None
                conventional_data = None

            # 提取晶格参数（用于主显示）
            lattice = display_struct.lattice
            lattice_params = {
                "a": round(lattice.a, 6),
                "b": round(lattice.b, 6),
                "c": round(lattice.c, 6),
                "alpha": round(lattice.alpha, 6),
                "beta": round(lattice.beta, 6),
                "gamma": round(lattice.gamma, 6)
            }

            # 提取原子位置（使用笛卡尔坐标用于主显示）
            atoms = []
            for site in display_struct:
                atoms.append({
                    "element": site.species_string,
                    "position": [round(x, 6) for x in site.coords.tolist()],  # 使用笛卡尔坐标
                    "occupancy": 1.0
                })

            result = {
                "id": structure_id,
                "name": filename.replace('.cif', ''),
                "formula": formula,
                "source": {
                    "database": "Generated",
                    "generator": "CrystaLLM",
                    "materialId": structure_id
                },
                "spaceGroup": space_group,
                "cifContent": cif_content,  # 统一使用 cifContent 字段
                "latticeParameters": lattice_params,
                "atoms": atoms,
                "properties": {
                    "density": float(display_struct.density),
                    "volume": float(display_struct.lattice.volume),
                    "numAtoms": len(atoms),
                    "spaceGroupNumber": space_group_number,
                    "crystalSystem": crystal_system
                },
                "metadata": {
                    "generation_id": generation_id,
                    "composition": composition,
                    "source": "crystallm",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # 添加cellTypes数据（如果成功分析了原胞和惯胞）
            if primitive_data and conventional_data:
                result["cellTypes"] = {
                    "primitive": primitive_data,
                    "conventional": conventional_data
                }
                result["currentCellType"] = "primitive"
                logger.info(f"✅ Added cellTypes data: primitive ({primitive_data['numAtoms']} atoms) and conventional ({conventional_data['numAtoms']} atoms)")

            return result
            
        except Exception as parse_error:
            logger.warning(f"Could not parse CIF with pymatgen: {parse_error}, returning basic structure")
            # 如果解析失败，返回基本结构
            return {
                "id": structure_id,
                "name": filename.replace('.cif', ''),
                "formula": formula,
                "source": {
                    "database": "Generated",
                    "generator": "CrystaLLM",
                    "materialId": structure_id
                },
                "spaceGroup": space_group,
                "cifContent": cif_content,  # 统一使用 cifContent 字段
                "metadata": {
                    "generation_id": generation_id,
                    "composition": composition,
                    "source": "crystallm",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
    except Exception as e:
        logger.error(f"Failed to convert CIF to frontend format: {e}")
        return None

if __name__ == "__main__":
    main()
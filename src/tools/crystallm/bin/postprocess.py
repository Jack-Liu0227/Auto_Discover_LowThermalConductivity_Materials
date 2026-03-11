# 导入所需的Python标准库
import os  # 用于文件和目录操作
import argparse  # 用于命令行参数解析
import io  # 用于处理字节流
import tarfile  # 用于处理tar压缩文件
import sys  # 用于修改Python路径

# 确保使用当前目录的CrystaLLM
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 从crystallm模块导入所需的函数
from crystallm import (
    extract_space_group_symbol,  # 提取空间群符号
    replace_symmetry_operators,  # 替换对称操作符
    remove_atom_props_block,  # 移除原子属性块
)




def postprocess(cif: str, fname: str) -> str:
    """
    对CIF文件内容进行后处理
    
    Args:
        cif: CIF文件的内容字符串
        fname: CIF文件名，用于错误报告
        
    Returns:
        处理后的CIF文件内容字符串
    """
    try:
        # 替换对称操作符为正确的操作符
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)

        # 移除原子属性块
        cif = remove_atom_props_block(cif)
    except Exception as e:
        # 如果处理过程中出现错误，在文件开头添加警告信息
        cif = "# WARNING: CrystaLLM could not post-process this file properly!\n" + cif
        print(f"error post-processing CIF file '{fname}': {e}")

    return cif


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Post-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the directory or .tar.gz file containing the "
                             "raw CIF files to be post-processed.")
    parser.add_argument("out", type=str,
                        help="Path to the directory or .tar.gz file where the "
                             "post-processed CIF files should be written")

    args = parser.parse_args()

    input_path = args.name
    output_path = args.out

    # 处理压缩文件的情况
    if input_path.endswith(".tar.gz"):
        with tarfile.open(input_path, "r:gz") as tar, tarfile.open(output_path, "w:gz") as out_tar:
            # 遍历压缩包中的所有文件
            for member in tar.getmembers():
                # 只处理.cif后缀的文件
                if member.isfile() and member.name.endswith(".cif"):
                    # 读取并解码文件内容
                    file = tar.extractfile(member)
                    cif_str = file.read().decode()
                    processed_cif = postprocess(cif_str, member.name)

                    # 将处理后的内容写入新的压缩文件
                    processed_file = io.BytesIO(processed_cif.encode())
                    tarinfo = tarfile.TarInfo(name=member.name)
                    tarinfo.size = len(processed_cif.encode())
                    out_tar.addfile(tarinfo, fileobj=processed_file)

                    print(f"processed: {member.name}")

    # 处理普通目录的情况
    else:
        # 如果输出目录不存在，创建它
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 遍历输入目录中的所有文件
        for filename in os.listdir(input_path):
            # 只处理.cif后缀的文件
            if filename.endswith(".cif"):
                # 读取文件内容
                file_path = os.path.join(input_path, filename)
                with open(file_path, "r") as file:
                    cif_str = file.read()
                    processed_cif = postprocess(cif_str, filename)

                # 将处理后的内容写入新文件
                output_file_path = os.path.join(output_path, filename)
                with open(output_file_path, "w") as file:
                    file.write(processed_cif)
                print(f"processed: {filename}")

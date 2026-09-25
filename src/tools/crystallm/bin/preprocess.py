import argparse
import gzip
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty

from crystallm import (
    semisymmetrize_cif,
    replace_data_formula_with_nonreduced_formula,
    add_atomic_props_block,
    round_numbers,
    extract_formula_units,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告


def progress_listener(queue, n):
    """监听进度队列,显示进度条"""
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def augment_cif(progress_queue, task_queue, result_queue, oxi, decimal_places):
    """处理CIF文件的主要函数
    
    Args:
        progress_queue: 进度队列,用于显示进度条
        task_queue: 任务队列,包含待处理的CIF文件
        result_queue: 结果队列,存储处理后的CIF文件
        oxi: 是否包含氧化态信息
        decimal_places: 小数点保留位数
    """
    augmented_cifs = []

    while not task_queue.empty():
        try:
            id, cif_str = task_queue.get_nowait()
        except Empty:
            break

        try:
            formula_units = extract_formula_units(cif_str)
            # 排除公式单位(Z)为0的CIF文件,这些是错误的
            if formula_units == 0:
                raise Exception()

            # 依次进行CIF文件的处理步骤
            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)  # 替换为非约化分子式
            cif_str = semisymmetrize_cif(cif_str)  # 对称化处理
            cif_str = add_atomic_props_block(cif_str, oxi)  # 添加原子属性
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)  # 数字四舍五入
            augmented_cifs.append((id, cif_str))
        except Exception:
            pass

        progress_queue.put(1)

    result_queue.put(augmented_cifs)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Pre-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be pre-processed. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--out", "-o", action="store",
                        required=True,
                        help="Path to the file where the pre-processed CIFs will be stored. "
                             "The file will contain the gzipped contents of a pickle dump. It is "
                             "recommended that the filename end in `.pkl.gz`.")
    parser.add_argument("--oxi", action="store_true",
                        help="Include this flag if the CIFs to be processed contain oxidation state information.")
    parser.add_argument("--decimal-places", type=int, default=4,
                        help="The number of decimal places to round the floating point numbers to in the CIF. "
                             "Default is 4.")
    parser.add_argument("--workers", type=int, default=4,
                        help="The number of workers to use for processing. Default is 4.")

    args = parser.parse_args()

    # 获取命令行参数
    cifs_fname = args.name
    out_fname = args.out
    oxi = args.oxi
    decimal_places = args.decimal_places
    workers = args.workers

    print(f"loading data from {cifs_fname}...")  # 加载数据
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    # 初始化多进程管理器和队列
    manager = mp.Manager()
    progress_queue = manager.Queue()  # 进度队列
    task_queue = manager.Queue()      # 任务队列
    result_queue = manager.Queue()    # 结果队列

    # 将所有CIF文件放入任务队列
    for id, cif in cifs:
        task_queue.put((id, cif))

    # 创建进度监听进程
    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(cifs),))

    # 创建工作进程
    processes = [mp.Process(target=augment_cif, args=(progress_queue, task_queue, result_queue, oxi, decimal_places))
                 for _ in range(workers)]
    processes.append(watcher)

    # 启动所有进程
    for process in processes:
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    # 收集处理结果
    modified_cifs = []
    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"number of CIFs: {len(modified_cifs)}")  # 输出处理后的CIF文件数量

    print(f"saving data to {out_fname}...")  # 保存处理后的数据
    with gzip.open(out_fname, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)

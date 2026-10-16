import lzma
import os
import math
from collections import defaultdict
from tabulate import tabulate

def compress_with_lzma(data):
    compressed_data = lzma.compress(data.encode('utf-8'))
    return len(compressed_data)

def calculate_compression_difference(training_data, test_slice):
    combined_data = training_data + test_slice
    combined_size = compress_with_lzma(combined_data)
    training_size = compress_with_lzma(training_data)
    return combined_size - training_size

class CramerVCalculator:
    """Computes Cramér's V coefficient and related statistics from a confusion matrix."""

    @staticmethod
    def compute(confusion_dict):
        authors = sorted(confusion_dict.keys())
        matrix = [
            [confusion_dict[row].get(col, 0) for col in authors]
            for row in authors
        ]

        n = sum(sum(row) for row in matrix)
        rows = len(matrix)
        cols = len(matrix[0])

        row_totals = [sum(row) for row in matrix]
        col_totals = [sum(matrix[i][j] for i in range(rows)) for j in range(cols)]

        chi2 = 0.0
        for i in range(rows):
            for j in range(cols):
                expected = (row_totals[i] * col_totals[j]) / n
                if expected > 0:
                    chi2 += (matrix[i][j] - expected) ** 2 / expected

        df = (rows - 1) * (cols - 1)
        cramers_v = math.sqrt(chi2 / (n * min(rows - 1, cols - 1)))

        diagonal_sum = sum(matrix[i][i] for i in range(min(rows, cols)))
        accuracy = diagonal_sum / n if n > 0 else 0.0

        return {
            "matrix": matrix,
            "authors": authors,
            "n": n,
            "rows": rows,
            "cols": cols,
            "row_totals": row_totals,
            "col_totals": col_totals,
            "chi2": chi2,
            "df": df,
            "cramers_v": cramers_v,
            "accuracy": accuracy,
            "diagonal_sum": diagonal_sum,
        }


def print_statistics(stats):
    print("\n" + "=" * 60)
    print("统计诊断")
    print("=" * 60)

    print(f"总样本数 n            : {stats['n']}")
    print(f"矩阵大小              : {stats['rows']} × {stats['cols']}")
    print(f"自由度 df             : {stats['df']}")
    print(f"卡方值 χ²             : {stats['chi2']:.4f}")
    print(f"Cramér's V            : {stats['cramers_v']:.4f}")
    print(f"正确分类数（对角线和）: {stats['diagonal_sum']}")
    print(f"总体准确率            : {stats['accuracy']:.4f} ({stats['accuracy']*100:.1f}%)")

    print("\n【行总计】每个测试作者的实际切片数：")
    max_row = max(stats["row_totals"]) if stats["row_totals"] else 0
    for author, total in zip(stats["authors"], stats["row_totals"]):
        flag = "" if total == max_row else "  ← 不足"
        print(f"  {author:<12}: {total}{flag}")

    print("\n【列总计】每个训练作者被预测的次数（预测分布）：")
    for author, total in zip(stats["authors"], stats["col_totals"]):
        ratio = total / stats["n"] * 100
        bar = "█" * int(ratio / 2)
        print(f"  {author:<12}: {total:>3}  ({ratio:5.1f}%)  {bar}")

    print("\n【召回率】每个测试作者被正确识别的比例：")
    for i, author in enumerate(stats["authors"]):
        correct = stats["matrix"][i][i]
        total = stats["row_totals"][i]
        recall = correct / total if total > 0 else 0.0
        print(f"  {author:<12}: {correct:>2}/{total:<2} = {recall*100:5.1f}%")

    print("=" * 60 + "\n")


def classify_slices(training_data_map, test_slices):
    result_table = defaultdict(lambda: defaultdict(int))
    for test_author, slices in test_slices.items():
        for test_slice in slices:
            best_match = None
            smallest_difference = float('inf')
            for train_author, training_data in training_data_map.items():
                difference = calculate_compression_difference(training_data, test_slice)
                if difference < smallest_difference:
                    smallest_difference = difference
                    best_match = train_author
            result_table[test_author][best_match] += 1
    return result_table

def display_result_table(result_table):
    authors = list(result_table.keys())
    headers = [""] + authors
    rows = []
    for test_author in authors:
        row = [test_author] + [result_table[test_author].get(train_author, 0) for train_author in authors]
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def load_text(file_path, size_in_kb):
    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(size_in_kb * 1024)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def load_test_slices(file_path, num_slices=32, slice_size_kb=2):
    text = load_text(file_path, num_slices * slice_size_kb)
    slice_size = slice_size_kb * 1024
    return [text[i * slice_size:(i + 1) * slice_size] for i in range(num_slices) if i * slice_size < len(text)]

def diagnose_files(training_files, test_files, train_kb, num_slices, slice_kb):
    print("\n" + "=" * 60)
    print(f"文件诊断（切片大小 {slice_kb}K，每文件 {num_slices} 片，总读取 {num_slices * slice_kb}K）")
    print("=" * 60)

    print(f"\n【训练文件】设定读取 {train_kb}K 字符（应读 {train_kb * 1024} 字符）")
    print(f"{'作者':<12} {'实际字符数':<14} {'约KB':<10} {'状态':<10}")
    print("-" * 50)
    for author, path in training_files.items():
        text = load_text(path, train_kb)
        actual = len(text)
        expected = train_kb * 1024
        kb = actual / 1024
        status = "✅ 满足" if actual >= expected else f"⚠️ 少 {expected - actual}"
        print(f"{author:<12} {actual:<14} {kb:<10.1f} {status:<10}")

    print(f"\n【测试文件】设定 {num_slices} 片 × {slice_kb}K = 需 {num_slices * slice_kb * 1024} 字符")
    print(f"{'作者':<12} {'实际字符数':<14} {'约KB':<10} {'实际片数':<10} {'状态':<10}")
    print("-" * 60)
    for author, path in test_files.items():
        text = load_text(path, num_slices * slice_kb)
        actual = len(text)
        kb = actual / 1024
        slices = load_test_slices(path, num_slices, slice_kb)
        n_slices = len(slices)
        status = "✅ 满" if n_slices == num_slices else f"⚠️ 少 {num_slices - n_slices}"
        print(f"{author:<12} {actual:<14} {kb:<10.1f} {n_slices:<10} {status:<10}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    BASE = r"C:\Users\han\Downloads\Assessing-the-quality-of-some-machine-translation-systems-using-an-information-theoretic-RS-method-main"

    # 翻译器列表：键为显示名称，值为文件名中的标识（小写）
    # 当前只启用 Google 和 Microsoft，其他已注释
    translators = {
        # "clude":     "clude",
         "Google":    "google",
        # "Deepseek":  "deepseek",
        # "Gimini":    "gemini",
         "Microsoft": "microsoft",
         "GPT":       "gpt",
    }

    def build_files(book, folder):
        """构建指定书（B1/B2）在指定文件夹（traning/test）下的文件字典"""
        files = {}
        for key, suffix in translators.items():
            # 命名规则：书编号-翻译器标识，例如 B1-google.txt
            filename = f"{book}-{suffix}.txt"
            files[key] = BASE + f"\\A-E\\{folder}\\" + filename
        return files

    TRAIN_KB = 110  # 训练读取量（K 字符）

    # 两本书，三种切片大小
    for book in ["B1", "B2"]:
        training_files = build_files(book, "traning")
        test_files = build_files(book, "test")

        for slice_kb in [1, 2, 4]:
            num_slices = 64 // slice_kb

            # ---------- 实验表头 ----------
            print("\n" + "#" * 80)
            print(f"# 实验：{book} 书")
            print(f"# 切片大小：{slice_kb}K    切片数量：{num_slices}    总读取量：{num_slices * slice_kb}K 字符")
            print(f"# 训练读取量：{TRAIN_KB}K 字符    翻译器：{', '.join(translators.keys())}")
            print("#" * 80)

            # 诊断
            diagnose_files(training_files, test_files, TRAIN_KB, num_slices, slice_kb)

            # 加载训练数据
            training_data_map = {
                author: load_text(path, TRAIN_KB) for author, path in training_files.items()
            }

            # 加载测试切片
            test_slices = {
                author: load_test_slices(path, num_slices, slice_kb)
                for author, path in test_files.items()
            }

            # 分类
            result_table = classify_slices(training_data_map, test_slices)

            # 输出结果
            print(f"\n{book} 书 - {slice_kb}K 切片分类结果：")
            display_result_table(result_table)

            stats = CramerVCalculator.compute(result_table)
            print_statistics(stats)
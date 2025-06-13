import random
from sympy import symbols, simplify, expand, cos, sin, Expr
from sympy.parsing.sympy_parser import parse_expr
from typing import Union
from multiprocessing import Process, Queue
import os
import time

def is_expression_equivalent_worker(
    expr1: str,
    expr2: str,
    variable: str,
    symbolic_check: bool,
    numeric_check: bool,
    num_samples: int,
    tolerance: float,
    sample_range: tuple,
    result_queue: Queue
) -> None:
    """
    工作进程函数，用于在单独的进程中执行表达式等价性检查。
    """
    try:
        x = symbols(variable)
        expr1_parsed = parse_expr(expr1.replace('^', '**'))
        expr2_parsed = parse_expr(expr2.replace('^', '**'))

        simplified1, simplified2 = expr1_parsed, expr2_parsed
        if symbolic_check:
            simplified1 = simplify(expand(expr1_parsed))
            simplified2 = simplify(expand(expr2_parsed))
            if simplified1.equals(simplified2):
                result_queue.put(True)
                return

        if numeric_check:
            for _ in range(num_samples):
                val = random.uniform(*sample_range)
                try:
                    res1 = simplified1.subs(x, val).evalf()
                    res2 = simplified2.subs(x, val).evalf()
                    if abs(res1 - res2) > tolerance:
                        result_queue.put(False)
                        return
                except:
                    continue

        result_queue.put(True)  # True if symbolic check passed or numeric check didn't fail
    except Exception as e:
        print(f"Error in worker: {e}")
        result_queue.put("error")  # Error

def is_expression_equivalent(
    expr1: str,
    expr2: str,
    variable: str = 'x',
    symbolic_check: bool = True,
    numeric_check: bool = True,
    num_samples: int = 10,
    tolerance: float = 1e-9,
    sample_range: tuple = (-1, 1),
    timeout: int = 20  # 超时时间（秒）
) -> Union[bool, str]:
    """
    判断两个数学表达式是否等价，支持超时功能。
    """
    result_queue = Queue()
    process = Process(target=is_expression_equivalent_worker, args=(
        expr1, expr2, variable, symbolic_check, numeric_check, num_samples, tolerance, sample_range, result_queue
    ))
    process.start()
    start_time = time.time()

    while time.time() - start_time < timeout:
        if not result_queue.empty():
            result = result_queue.get()
            process.terminate()
            process.join()
            return result
        time.sleep(0.1)  # 每隔 0.1 秒检查一次

    # 如果超时，强制终止子进程
    process.terminate()
    process.join()
    return "timeout"

# 示例用法
if __name__ == "__main__":
    expr1 = "x^2 + 2*x + 1"
    expr2 = "(x + 1)^2"
    result = is_expression_equivalent(expr1, expr2, timeout=5)
    print(result)  # 输出：True 或 False 或 "timeout" 或 "error"
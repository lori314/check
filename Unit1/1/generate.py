import random
import string

def generate_integer():
    sign = random.choice(['+', '-', ''])
    num = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 2)))
    return sign + num

def generate_factor(allow_params):
    if random.random() < 0.5 and allow_params:
        return random.choice(allow_params)
    else:
        return generate_integer()

def generate_power_expression(base):
    exponent = random.randint(2, 4)
    return f"({base})^{exponent}"

def generate_expression(depth=0, allow_params=None):
    if depth > 1:
        factor = generate_factor(allow_params)
        return factor if random.random() < 0.3 else generate_power_expression(factor)
    operators = ['+', '-', '*']
    expr = generate_factor(allow_params)
    for _ in range(random.randint(1, 2)):
        operator = random.choice(operators)
        next_expr = generate_expression(depth + 1, allow_params)
        if operator == '*' and next_expr.startswith(('+', '-')):
            next_expr = f"({next_expr})"
        expr += f"{operator}{next_expr}"
    if random.random() < 0.3:
        expr = generate_power_expression(f"({expr})")
    return expr

def generate_trig_function(allow_params):
    func = random.choice(['sin', 'cos'])
    arg = generate_expression(allow_params=allow_params)
    if random.random() < 0.5:
        return generate_power_expression(f"{func}(({arg}))")
    else:
        return f"{func}(({arg}))"

def generate_function_definition(params):
    allow_params = params.copy()
    f0 = f"f{{0}}({','.join(params)}) = {generate_expression(allow_params=allow_params)}"
    f1 = f"f{{1}}({','.join(params)}) = {generate_expression(allow_params=allow_params)}"
    fn = f"f{{n}}({','.join(params)}) = {generate_integer()}*f{{n-1}}({','.join(params)}) + {generate_integer()}*f{{n-2}}({','.join(params)}) + {generate_expression(allow_params=allow_params)}"
    return [f0, f1, fn]

def generate_sample(n):
    if n == 0:
        expr = generate_expression(allow_params=['x'])
        if random.random() < 0.5:
            expr += f"+{generate_trig_function(['x'])}"
        return ["0"], expr
    else:
        params = random.choice([['x'], ['y'], ['x', 'y']])
        defs = generate_function_definition(params)
        expr = generate_expression(allow_params=['x'])
        call_args = ['('+ generate_expression(allow_params=['x'])+')' for _ in params]
        expr += f"+f{{{random.randint(0,3)}}}({','.join(call_args)})"
        return ["1"] + defs, expr

def generate_test():
# 生成测试样例
    n = random.choice([0,1]) # 示例输入n=1
    input_data, expression = generate_sample(n)
    res = ""
    for line in input_data:
        res = res + line + "\n"
    res = res + expression
    return res

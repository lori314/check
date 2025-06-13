import random
import string

def generate_integer():
    sign = random.choice(['+', '-', ''])
    num = ''.join(random.choice(string.digits) for _ in range(random.randint(1, 2)))
    return sign + num

def generate_factor(allow_params):
    if random.random() < 0.8 and allow_params:
        return random.choice(allow_params)
    else:
        return generate_integer()
    
def generate_com_factor(depth,allow_params):
    global func_list
    r = random.random()
    if r < 0.2:
        return generate_derivative(depth,allow_params=allow_params)
    elif 0.5 < r and r < 0.9 and len(func_list)!=0:
        return generate_fuction(depth,allow_params)
    elif 0.2 < r and r < 0.4:
        return generate_trig_function(depth,allow_params=allow_params)
    elif r > 0.4 and r < 0.5:
        return generate_expression(depth,allow_params=allow_params)
    else:
        return generate_factor(allow_params=allow_params)

def generate_com_factor_withoutd(depth,allow_params):
    global func_list
    r = random.random()
    if 0.5 < r and r < 0.9 and len(func_list)!=0:
        return generate_fuction(depth,allow_params)
    elif 0.2 < r and r < 0.4:
        return generate_trig_function_withoutd(depth,allow_params=allow_params)
    elif r > 0.4 and r < 0.5:
        return generate_expression_withoutd(depth,allow_params=allow_params)
    else:
        return generate_factor(allow_params=allow_params)

def generate_power_expression(base):
    exponent = random.randint(2, 4)
    return f"({base})^{exponent}"

def generate_expression_withoutd(depth=0,allow_params=None):
    if depth > 1:
        factor = generate_factor(allow_params)
        return factor if random.random() < 0.3 else generate_power_expression(factor)
    operators = ['+', '-', '*']
    expr = generate_com_factor_withoutd(depth + 1,allow_params)
    for _ in range(random.randint(1, 2)):
        operator = random.choice(operators)
        next_expr = generate_expression_withoutd(depth + 1, allow_params)
        if operator == '*' and next_expr.startswith(('+', '-')):
            next_expr = f"({next_expr})"
        expr += f"{operator}{next_expr}"
    if random.random() < 0.3:
        expr = generate_power_expression(f"({expr})")
    return expr

def generate_expression(depth=0, allow_params=None):
    if depth > 1:
        factor = generate_factor(allow_params)
        return factor if random.random() < 0.3 else generate_power_expression(factor)
    operators = ['+', '-', '*']
    expr = generate_com_factor(depth + 1,allow_params)
    for _ in range(random.randint(1, 2)):
        operator = random.choice(operators)
        next_expr = generate_expression(depth + 1, allow_params)
        if operator == '*' and next_expr.startswith(('+', '-')):
            next_expr = f"({next_expr})"
        expr += f"{operator}{next_expr}"
    if random.random() < 0.3:
        expr = generate_power_expression(f"({expr})")
    return expr

def generate_trig_function_withoutd(depth,allow_params):
    func = random.choice(['sin', 'cos'])
    arg = generate_expression_withoutd(depth,allow_params=allow_params)
    if random.random() < 0.5:
        return generate_power_expression(f"{func}(({arg}))")
    else:
        return f"{func}(({arg}))"

def generate_trig_function(depth,allow_params):
    func = random.choice(['sin', 'cos'])
    arg = generate_expression(depth,allow_params=allow_params)
    if random.random() < 0.5:
        return generate_power_expression(f"{func}(({arg}))")
    else:
        return f"{func}(({arg}))"
    
def generate_fuction(depth,allow_params):
    global func_list
    func = random.choice(func_list)
    call_args = ['('+ generate_expression(depth+1,allow_params=allow_params)+')' for _ in range(func[1])]
    if func[0] == 'f' :
        return f"f{{{random.randint(0,3)}}}({','.join(call_args)})"
    else:
        return f"{func[0]}({','.join(call_args)})"
    
def generate_derivative(depth,allow_params):
    return f"dx(({generate_expression(depth,allow_params=allow_params)}))"

def generate_function_definition(params):
    global func_list
    allow_params = params.copy()
    f0 = f"f{{0}}({','.join(params)}) = {generate_expression_withoutd(depth = 2,allow_params=allow_params)}"
    f1 = f"f{{1}}({','.join(params)}) = {generate_expression_withoutd(depth = 2, allow_params=allow_params)}"
    fn = f"f{{n}}({','.join(params)}) = {generate_integer()}*f{{n-1}}({','.join(params)}) + {generate_integer()}*f{{n-2}}({','.join(params)}) + {generate_expression_withoutd(allow_params=allow_params)}"
    func_list.append(('f',len(allow_params)))
    return [f0,f1,fn]

def generate_function_easy(params):
    global func_list
    allow_params = params.copy()
    if len(func_list)!=0:
        func = 'g'
    else:
        func = 'h'
    f = f"{func}({','.join(params)}) = {generate_expression_withoutd(depth=1,allow_params=allow_params)}"
    func_list.append((f"{func}",len(params)))
    return f

def generate_sample(m,n):
    global func_list
    func_list = []
    input_data = []
    input_data = input_data + [str(n)]
    for i in range(n):
        params = random.choice([['x'],['y'],['x','y']])
        defs = generate_function_easy(params)
        input_data.append(defs)
    
    input_data = input_data + [str(m)]
    for i in range(m):
        params = random.choice([['x'], ['y'], ['x', 'y']])
        defs = generate_function_definition(params)
        input_data = input_data + defs 

    expr = generate_expression(allow_params=['x'])
    return input_data, expr

    # if m == 0:
    #     expr = generate_expression(allow_params=['x'])
    #     if random.random() < 0.5:
    #         expr += f"+{generate_trig_function(['x'])}"
    #     return ["0"], expr
    # else:
    #     params = random.choice([['x'], ['y'], ['x', 'y']])
    #     defs = generate_function_definition(params)
    #     expr = generate_expression(allow_params=['x'])
    #     call_args = ['('+ generate_expression(allow_params=['x'])+')' for _ in params]
    #     expr += f"+f{{{random.randint(0,3)}}}({','.join(call_args)})"
    #     return ["1"] + defs, expr

def generate_test():
# 生成测试样例
    m = random.choice([0,1]) # 示例输入n=1
    n = random.choice([0,1,2])
    input_data, expression = generate_sample(m,n)
    res = ""
    for line in input_data:
        res = res + line + "\n"
    res = res + expression
    return res

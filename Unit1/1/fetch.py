import subprocess
import os

def run_bat_safely(bat_path, input_data):
    """安全执行 .bat 文件并捕获错误"""
    try:
        # 验证路径和权限
        if not os.path.exists(bat_path):
            raise FileNotFoundError(f"文件不存在: {bat_path}")
        if not os.access(bat_path, os.X_OK):
            raise PermissionError(f"无执行权限: {bat_path}")

        # 执行并传递输入
        with subprocess.Popen(
            bat_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True  # 确保命令行扩展生效
        ) as proc:
            stdout, stderr = proc.communicate(input=input_data)
            
            # 检查返回码
            if proc.returncode != 0:
                print(f"[错误] 执行 {bat_path} 失败，返回码: {proc.returncode}")
                print(f"[错误信息]\n{stderr}")
                return None
            return stdout
    except Exception as e:
        print(f"[异常] {str(e)}")
        return None

# 示例用法
if __name__ == "__main__":
    # 第一个 .bat 文件
    bat1 = r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\1\lty.bat"
    input1 = """1
f{0}(x,y) = 94-+26+3+y-7*(-52+(-14)^4-(+93)^3)
f{1}(x,y) = -4*y+((x-(x)^3))^2++6+(y)^4--33+19*(-2)*(x)^2
f{n}(x,y) = 2*f{n-1}(x,y) + -34*f{n-2}(x,y) + ((-8-((x+-87*(x)^3-+0+((+30+y))^2))^4*00--01+-3))^3
x-((+75+((+31*(+42)^4*(x)^2))^2-((+5-(+15)^2))^2))^2+-7*((+81*x))^2+f{5}(+63*x*x+(x)^3-(+38)^2,-6-((8+55*x*(+47)^3))^2-94-x*(6)^4)"""
    output1 = run_bat_safely(bat1, input1)
    if output1 is not None:
        print(f"第一个脚本输出:\n{output1}")

    # 第二个 .bat 文件
    bat2 = r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\1\sqh.bat"
    input2 = "0\nx+x"
    output2 = run_bat_safely(bat2, input2)
    if output2 is not None:
        print(f"第二个脚本输出:\n{output2}")
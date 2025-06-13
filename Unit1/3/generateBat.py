import os
def create_bat_file(jar_path, bat_name, output_dir=None):
    """
    为给定的 JAR 文件生成一个 .bat 文件，并返回该文件的路径。
    :param jar_path: JAR 文件的路径
    :param output_dir: 输出目录（可选，默认为当前目录）
    :return: 生成的 .bat 文件的路径
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成 .bat 文件名
    bat_file_name = bat_name
    bat_path = os.path.join(output_dir, bat_file_name)
    
    # 写入 .bat 文件内容
    with open(bat_path, 'w') as bat_file:
        bat_file.write("@echo off\n")
        
        bat_file.write("REM 设置 JAR 文件的路径\n")
        bat_file.write(f"set JAR_PATH={jar_path}\n\n")
        
        bat_file.write("REM 启动 JAR 文件\n")
        bat_file.write("java -jar \"%JAR_PATH%\"\n\n")
    
    print(f"Generated .bat file: {bat_path}")
    return bat_path

# 示例：为一个 JAR 文件生成 .bat 文件
jar_path = r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\3\jars\lzk3.jar"
print(create_bat_file(jar_path,"lzk3.bat"))

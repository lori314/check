import jpype

class JAR(object):
    def __init__(self,jar_path) -> None:
        self.jar_path=jar_path

    def call_jar_main(self,input_data):
        jar_path = self.jar_path
        # 启动 JVM 并设置类路径
        jvm_path = jpype.getDefaultJVMPath()
        jpype.startJVM(jvm_path, "-Djava.class.path=" + jar_path)

        # 将输入数据转换为字节数组
        input_bytes = input_data.encode("utf-8")
        ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")
        ByteArrayOutputStream = jpype.JClass("java.io.ByteArrayOutputStream")
        PrintStream = jpype.JClass("java.io.PrintStream")
        System = jpype.JClass("java.lang.System")

        # 设置 Java 的 System.in 为 ByteArrayInputStream
        System.setIn(ByteArrayInputStream(input_bytes))

        # 捕获 System.out 和 System.err
        original_out = System.out
        original_err = System.err
        output_stream = ByteArrayOutputStream()
        error_stream = ByteArrayOutputStream()
        System.setOut(PrintStream(output_stream))
        System.setErr(PrintStream(error_stream))

        # 加载 Java 的 JarFile 类
        JarFile = jpype.JClass("java.util.jar.JarFile")
        File = jpype.JClass("java.io.File")

        # 打开 JAR 文件
        jar_file = JarFile(File(jar_path))

        # 遍历 JAR 文件中的条目，查找包含 main 方法的类
        entries = jar_file.entries()
        found_main = False
        while entries.hasMoreElements():
            entry = entries.nextElement()
            entry_name = str(entry.getName())  # 转换为 Python 字符串
            if entry_name.endswith(".class"):
                class_name = entry_name.replace("/", ".").replace(".class", "")
                try:
                    java_class = jpype.JClass(class_name)
                    if hasattr(java_class, "main"):
                        java_class.main(jpype.JArray(jpype.JString)([]))  # 调用 main 方法
                        found_main = True
                        break
                except Exception as e:
                    continue

        # 恢复 System.out 和 System.err
        System.setOut(original_out)
        System.setErr(original_err)

        # 获取输出内容
        output = str(output_stream.toString("UTF-8"))
        error = str(error_stream.toString("UTF-8"))

        # 关闭 JVM
        jpype.shutdownJVM()

        # 返回输出内容
        return output, error

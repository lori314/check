from Program import Program
import generate
import isSame
import os

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # 初始化 Program 对象
    my_jar = Program(r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\3\lty2.bat")
    his_jar = Program(r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\3\lzk3.bat")



    # # 打印测试数据和输出
    # print("测试数据：")
    # print(test)
    # print("我的输出：")
    # print(my_res)

    # print("="*50)
    # print("他的输出：")
    # print(his_res)

    # # 比较输出是否一致
    # same = isSame.is_expression_equivalent(my_res, his_res)
    # if same:
    #     print("success")
    # else:
    #     print("false")

    test_num = 300
    for i in range(test_num):
        # 生成测试数据
        test = generate.generate_test()
        with open("test.txt","w") as f:
            f.write(test)
        # 调用 .bat 文件并获取输出
        my_res = my_jar.call_bat(test)
        his_res = his_jar.call_bat(test)
        if his_res==None:
            print(test)
            break
        with open("out.txt","w") as fout:
            fout.write(his_res)

        print("test"+str(i)+"begin")
        my_len=len(my_res)
        his_len=len(his_res)
        print("my_len:"+str(my_len))
        print("his_len:"+str(his_len))
        # if abs(my_len-his_len)/max(my_len,his_len) > 0.2:
        #     print("长度差别过大")
        #     print("test"+str(i)+"false")
        #     print(test)
        #     print(my_res)
        #     print("="*50)
        #     print(his_res)
        #     break
        # elif abs(my_len-his_len) <= 1 and my_len > 60  and his_len>60:
        #     print("test"+str(i)+"success")
        #     continue
        if abs(my_len-his_len) <= 1 and my_len > 60  and his_len>60:
            print("test"+str(i)+"success")
            continue
        same = isSame.is_expression_equivalent(my_res, his_res)
        if same == "timeout":
            print("test"+str(i)+"timeout")
            continue
        if same:
            print("test"+str(i)+"success")
        else:
            print("test"+str(i)+"false")
            print(test)
            print(my_res)
            print("="*50)
            print(his_res)
            break
from Program import Program
import isSame
import os

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # 初始化 Program 对象
    my_jar = Program(r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\3\lty2.bat")
    his_jar = Program(r"C:\Users\lenovo\Desktop\codes\python\checkMachine\Unit1\3\tianji.bat")

    with open("test.txt","r") as f:
        test=f.read()
    my_res = my_jar.call_bat(test)
    his_res = his_jar.call_bat(test)
    if his_res==None:
        print(test)
    my_len=len(my_res)
    his_len=len(his_res)
    print("my_len:"+str(my_len))
    print("his_len:"+str(his_len))
    same = isSame.is_expression_equivalent(my_res, his_res)
    if same == "timeout":
        print("test"+"timeout")
    if same:
        print("test"+"success")
    else:
        print("test"+"false")
        print(test)
        print(my_res)
        print("="*50)
        print(his_res)
import fetch

class Program(object):
    def __init__(self,bat_path) -> None:
        self.bat_path = bat_path

    def call_bat(self,input_data):
        res = fetch.run_bat_safely(self.bat_path,input_data)
        return res
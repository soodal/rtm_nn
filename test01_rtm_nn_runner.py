import os
import time

if __name__ == '__main__':
    for i in range(100):
        os.system('python test01_rtm_nn.py')
        print('error occured. wait 10 seconds')
        time.sleep(10)

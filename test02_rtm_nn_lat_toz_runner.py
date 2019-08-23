import os
import time

if __name__ == '__main__':
    for i in range(100):
        os.system('python test02_rtm_nn_lat_toz.py')
        print('error occured. wait 3 seconds')
        time.sleep(3)

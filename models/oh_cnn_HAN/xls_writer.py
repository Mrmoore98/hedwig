# -*- coding: UTF-8 -*-
 
import time
from xlwt import *
from models.oh_cnn_HAN.args import get_args
from copy import deepcopy

def write_xls(train, dev, config, fname= 11):
    
    file = Workbook(encoding = 'utf-8')
    #指定file以utf-8的格式打开
    table = file.add_sheet('data')
    #指定打开的文件名
    

    
    table.write(0, 1, 'train acc')
    table.write(0, 2, 'train loss')
    for i, num in enumerate(train):
        table.write(i+1, 1, num[0])
        table.write(i+1, 2, num[1])
    

    table.write(0, 3, 'dev acc')
    table.write(0, 4, 'dev loss')
    table.write(0, 5, 'tot time')
    for i, name in enumerate(dev):
        table.write(i+1, 3, name[0])
        table.write(i+1, 4, name[1])
        table.write(i+1, 5, name[2])
    
    table.write(0, 6, 'Config: Key')
    table.write(0, 7, 'Config: Data')
    keys = [i  for i in dir(config) if i[0] != '_']
    for i, key in enumerate(keys):
        
        data =  getattr(config,key)
        if type(data) is list: 
            data = ','.join([str(i) for i in data])
            data = '[ '+ data +' ]'
        try: 
            table.write(i+1, 6, key)   
            table.write(i+1, 7, data)
        except:
            continue
        


    file.save('model_{}_results_{}.xls'.format(fname, time.asctime( time.localtime(time.time()) )))



if __name__ == "__main__":
    start_time = time.time()
    print("Start")
    aa = [(1,2),(2,3)]
    bb = []
    for i in range(10):
        bb.append([i,222,232323])
    args = get_args()
    config = deepcopy(args)
    config.output_channel  = [400, 400]
    config.input_channel   = 30000
    config.kernel_H        = [4, 3]
    config.kernel_W        = [2, 3]
    config.stride          = [1, 1]
    config.rnn_hidden_size = 300
    config.max_size        = 30000
    config.fill_value      = 5
    config.use_RNN         = True
    config.id              = 1
    config.hierarchical    = True
    config.attention       = True

    write_xls(aa, bb, config)
    print("Process Complete in {:3d}s!".format(int(time.time()-start_time)))
import csv
import os

def write_ohlcv(ohlcv_list, name, del_first=True):
    file_path = os.path.join('datasets', f'{name}.csv')
    
    if del_first and os.path.exists(file_path):
        os.remove(file_path)

    data_file = open(file_path, 'w+')
    
    # create the csv writer object
    print(f'writing: {data_file}')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])
    
    for ohlcv_item in ohlcv_list:
        csv_writer.writerow(ohlcv_item.list())
            
    data_file.close()
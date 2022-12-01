# pylint: disable=C0103,C0111,C0301

import argparse
import os
import sys
from os import path as osp

import numpy as np
import pandas
import scipy.interpolate

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))




def interpolate_vector_linear(input, input_timestamp, output_timestamp):
    """
    This function interpolate n-d vectors (despite the '3d' in the function name) into the output time stamps.

    Args:
        input: Nxd array containing N d-dimensional vectors.
        input_timestamp: N-sized array containing time stamps for each of the input quaternion.
        output_timestamp: M-sized array containing output time stamps.
    Return:
        quat_inter: Mxd array containing M vectors.
    """
    assert input.shape[0] == input_timestamp.shape[0]
    func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)#,kind=2)
    interpolated = func(output_timestamp)
    return interpolated


def process_data_source(raw_data, output_time, method):
    input_time = raw_data[:, 0]
    if method == 'vector':
        output_data = interpolate_vector_linear(raw_data[:, 1:], input_time, output_time)
    else:
        raise ValueError('Interpolation method must be "vector" or "quaternion"')
    return output_data


def compute_output_time(all_sources, sample_rate=50):
    """
    Compute the output reference time from all data sources. The reference time range must be within the time range of
    all data sources.
    :param data_all:
    :param sample_rate:
    :return:
    """
    interval = 1. / sample_rate
    min_t = max([data[0, 0] for data in all_sources.values()]) + interval
    max_t = min([data[-1, 0] for data in all_sources.values()]) - interval
    return np.arange(min_t, max_t, interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default="./lists/data_list.txt", help='Path to a list file.')
    # parser.add_argument('--path', type=str, default="./test_case0", help='Path to a dataset folder.')
    parser.add_argument('--path', type=str, default=None, help='Path to a dataset folder.')
    parser.add_argument('--skip_front', type=int, default=21, help='Number of discarded records at beginning.')
    parser.add_argument('--skip_end', type=int, default=21, help='Number of discarded records at end')
    parser.add_argument('--output_samplerate', type=int, default=50, help='Output sample rate. Default is 50Hz')
    # parser.add_argument('--input_folder', type=str, default="./TestSet/")
    # parser.add_argument('--output_folder', type=str, default="./TestSet/processed/")
    parser.add_argument('--input_folder', type=str, default="./data/raw/")
    parser.add_argument('--output_folder', type=str, default="./data/processed/")
    parser.add_argument('--mode', type=str, default='train', help='Have Location or not.')
    # parser.add_argument('--output_folder', type=str, default="./test_case0/processed/")

    args = parser.parse_args()

    dataset_list = []
    if args.path:
        args.input_folder = ''
        dataset_list.append(args.path)
    elif args.list:
        with open(args.list) as f:
            for s in f.readlines():
                if s[0] != '#':
                    dataset_list.append(s.strip('\n'))
    else:
        raise ValueError('No data specified')

    print(dataset_list)

    total_length = 0.0
    length_dict = {}
    for dataset in dataset_list:
        if len(dataset.strip()) == 0:
            continue
        if dataset[0] == '#':
            continue
        info = dataset.split(',')
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]
        data_root = args.input_folder + info[0]
        length = 0
        
        print('------------------\nProcessing ' + data_root, ', type: ' + motion_type)
        all_sources = {}
        source_vector = {'Gyroscope', 'Accelerometer', 'Linear Accelerometer', 'Magnetometer','Location'}
        source_quaternion = {}
        reference_time = 0
        source_all = source_vector.union(source_quaternion)
        is_input = ''
        for source in source_all:
            try:
                if args.mode == 'test' and source == 'Location':
                    is_input = '_input'
                else:
                    is_input = ''
                print(f'load {source+is_input}.csv')
                source_data = np.genfromtxt(osp.join(data_root, source+ is_input + '.csv'),dtype=float,delimiter =',')[1:]
                source_data[:, 0] = (source_data[:, 0] - reference_time) 
                all_sources[source] = np.array(pandas.DataFrame(source_data).drop_duplicates([0]))
            except OSError:
                print('Can not find file for source {}. Please check the dataset.'.format(osp.join(data_root, source+ is_input + '.csv'),dtype=float,delimiter =','))
                exit(1)
        for src_id, src in all_sources.items():
            print('Source: %s,  start time: %f, end time: %f' % (src_id, src[0, 0], src[-1, 0]))

        output_time = compute_output_time(all_sources, args.output_samplerate)
        if motion_type not in length_dict:
            length_dict[motion_type] = 0
        length_dict[motion_type] += output_time[-1] - output_time[0]
        # 去掉突变
        for i in range(1,len(all_sources['Location'][1:, [5]])):
            # print(all_sources['Location'][0,:])
            if(all_sources['Location'][1:, [5]][i-1] > 350 and all_sources['Location'][1:, [5]][i] < 10):
                print('=====')
                all_sources['Location'][1:, [5]][i] = 359
            elif(all_sources['Location'][1:, [5]][i-1] < 10 and all_sources['Location'][1:, [5]][i] > 350):
                print('-----')
                all_sources['Location'][1:, [5]][i] = 1


        processed_source = {}
        for source in all_sources.keys():
            raw_data = all_sources[source]
            input_sr = (raw_data.shape[0] - 1) / (raw_data[-1, 0] - raw_data[0, 0])
            if source in source_vector:
                processed_source[source] = process_data_source(all_sources[source], output_time, 'vector')
            print('{} found. Input sampling rate: {}Hz. Channel size:{}'.format(source, input_sr,
                                                                                processed_source[source].shape[1]))

        # construct a Pandas DataFrame
        column_list = 'time,' \
                        'gyro_x,gyro_y,gyro_z,' \
                        'acce_x,acce_y,acce_z,' \
                        'linacce_x,linacce_y,linacce_z,' \
                        'magnet_x,magnet_y,magnet_z,'\
                        'Location_delta_x,Location_delta_y,Location_dir'.split(',')
                        # {'Gyroscope', 'Accelerometer', 'Linear Accelerometer', 'Magnetometer', 'Barometer'}
        
        data_mat = np.concatenate([output_time[1:, None],
                                    processed_source['Gyroscope'][1:],
                                    processed_source['Accelerometer'][1:],
                                    processed_source['Linear Accelerometer'][1:],
                                    processed_source['Magnetometer'][1:],
                                    (processed_source['Location'][1:, [0, 1]]-processed_source['Location'][:-1, [0, 1]])*1e9,
                                    processed_source['Location'][1:, [4]]
                                    ], axis=1)
        data_mat = data_mat[args.skip_front:-args.skip_end]
        if not osp.isdir(args.output_folder):
            os.makedirs(args.output_folder)
        data_pandas = pandas.DataFrame(data_mat, columns=column_list)
        # print(data_pandas)
        if args.mode == 'train':
            data_pandas.drop(data_pandas[data_pandas.Location_dir < 0].index, inplace=True)
        data_pandas.to_csv(args.output_folder + f'{dataset}.csv')
        print('Dataset written to ' + args.output_folder + f'{dataset}.csv')
        # print(data_pandas)

        
    print('All done. Total length: {:.2f}s ({:.2f}min)'.format(total_length, total_length / 60.0))
    for k, v in length_dict.items():
        print(k + ': {:.2f}s ({:.2f}min)'.format(v, v / 60.0))

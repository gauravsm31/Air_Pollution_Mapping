
import csv
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

from matplotlib import rcParams
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score

def first_process_data(fname):
    df = pd.read_csv(fname)
    # fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    # for i, res in enumerate(['l', 'h']):
    #     m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2,
    #                 width=90000, height=120000, resolution=res, ax=ax[i])
    #     m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
    #     m.drawmapboundary(fill_color="#DDEEFF")
    #     m.drawcoastlines()
    #     ax[i].set_title("resolution='{0}'".format(res))
    # plt.show()

    fig = plt.figure(figsize=(12, 8))
    m = Basemap(projection='lcc', resolution='h',
                width=25000, height=20000, 
                lat_0=statistics.mean(df['latitude']), lon_0=statistics.mean(df['longitude']),)
    m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
    m.drawmapboundary(fill_color="#DDEEFF")
    m.drawcoastlines()
    for i in range(len(df['id'])):
        x, y = m(df['longitude'][i], df['latitude'][i])
        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x, y, str(df['id'][i]), fontsize=12)
    
    plt.show()
    # Map (long, lat) to (x, y) for plotting
    # x, y = m(-122.3, 47.6)
    # plt.plot(x, y, 'ok', markersize=5)
    # plt.text(x, y, ' Seattle', fontsize=12)


def gather_data(gen_file, monthes):
    seed_filenames = ["january-2017.csv","february-2017.csv","march-2017.csv","april-2017.csv","may-2017.csv","june-2017.csv","july-2017.csv","august-2017.csv","september-2017.csv","october-2017.csv","november-2017.csv","december-2017.csv"]
    filenames = []
    for month in monthes:
        filenames.append(seed_filenames[month - 1])
        print(seed_filenames[month - 1])
    path =r'./data/'

    li = []
    for filename in filenames:
        fname = path + filename 
        df = pd.read_csv(fname, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['UTC time'] = pd.to_datetime(frame['UTC time'])

    sens_test = [173,189,196,201,222,228]
    sens_test = str(sens_test)
    sens_train = [169,170,171,172,176,177,179,181,182,183,184,192,194,203,204,212,214,215,218,219,220,221,223,225,226,263]
    sens_train = str(sens_train)
    sens = sens_train + sens_test
    sens = str(sens)
    cols = [c for c in frame.columns if c.lower()[:3] in sens]
    cols = cols + ['UTC time']
    frame_coldrop = frame[cols]

    names = ['temp', 'hum', 'pres', 'pm25']
    stat_map = {}
    for name in names:
        stat_map[name] = [0.0, float("inf")]
        for col in frame_coldrop.filter(regex=name):
            stat_map[name][0] = max(frame_coldrop[col].max(), stat_map[name][0])
            if name == 'hum' and frame_coldrop[col].min() < 0: continue
            stat_map[name][1] = min(frame_coldrop[col].min(), stat_map[name][1])
            
    print(stat_map)
    for name in stat_map:
        stat_map[name][0] = stat_map[name][0] - stat_map[name][1]

    if not gen_file: return float(stat_map['pm25'][0])
    nan_count = 0
    overall_count = 0
    if gen_file:
        train_f = open("train.txt", "w")
        test_f = open("test.txt", "w")
        time = frame_coldrop['UTC time']
        bad_hum = 0
        for name in names:
            for cname in frame_coldrop.filter(regex=name):
                if cname.lower()[:3] in sens_train or (cname.lower()[:3] in sens_test and 'pm' not in name):
                    cur_col = frame_coldrop[cname]
                    for i in range(len(cur_col)):
                        overall_count += 1
                        if str(cur_col.iloc[i]).lower() == 'nan': nan_count += 1
                        if str(cur_col.iloc[i]).lower() != 'nan':
                            if name == 'hum' and int(cur_col.iloc[i]) < 0: 
                                continue
                            train_f.write(cname.lower()[:3] + "\t")
                            train_f.write(cname.lower()[4:] + "-")
                            train_f.write(str(time.iloc[i])[:10] + "-" + str(time.iloc[i])[11:13] + "\t")
                            train_f.write(str(round(float(cur_col.iloc[i] - stat_map[name][1]) * 1000.0 / float(stat_map[name][0]), 6)))
                            train_f.write('\n') 
                if cname.lower()[:3] in sens_test and 'pm25' in name:
                    cur_col = frame_coldrop[cname]
                    for i in range(len(cur_col)):
                        if str(cur_col.iloc[i]).lower() != 'nan':
                            if name == 'hum' and int(cur_col.iloc[i]) < 0: 
                                continue
                            test_f.write(cname.lower()[:3] + "\t")
                            test_f.write(cname.lower()[4:] + "-")
                            test_f.write(str(time.iloc[i])[:10] + "-" + str(time.iloc[i])[11:13] + "\t")
                            test_f.write(str(round(float(cur_col.iloc[i] - stat_map[name][1]) * 1000.0 / float(stat_map[name][0]), 6)))
                            test_f.write('\n') 
        print(nan_count, overall_count, nan_count * 1.0 / overall_count)            
        loc_data = pd.read_csv("./data/sensor_locations.csv")
        lats = loc_data['latitude'].tolist()
        lons = loc_data['longitude'].tolist()
        min_lat = min(lats)
        range_lat = round(max(lats) - min_lat, 6)
        min_lons = min(lons)
        range_lon = round(max(lons) - min_lons, 6)

        for num in range(1):
            for i in range(len(loc_data['id'])):
                loc_id = str(loc_data['id'][i])
                if len(loc_id) == 3 and loc_id in sens:
                    # loc_id = str(loc_id)
                    train_f.write(loc_id + "\t")
                    train_f.write("latitude" + str(num) + "\t")
                    train_f.write(str(round(round((lats[i] - min_lat), 6) * 1000.0 / range_lat, 6)) + "\n")
                    train_f.write(loc_id + "\t")
                    train_f.write("longitude" + str(num) +"\t")
                    train_f.write(str(round(round((lons[i] - min_lons), 6) * 1000.0 / range_lon, 6)) + "\n")

        train_f.close()
        test_f.close()
    # print(min_lat,max(lats), range_lat, min_lons, max(lons), range_lon)
    return float(0)

def compare_data(pm25_range):
    
    test = []
    pred_temp = []
    pred = []
    day_map = {}
    sensor_map = {}
    sens_test = ['173','189','196','201','222','228']
    with open("test.txt", mode='r') as f:
        line = f.readline()
        while line:
            u, i, s = line.split()
            if 'pm25' in i:
                day = i
                if day not in day_map:
                    day_map[day] = {}
                    day_map[day]["true"] = []
                    day_map[day]["pred"] = []
                day_map[day]["true"].append((u+i, s))

                if u not in sensor_map:
                    sensor_map[u] = {}
                    sensor_map[u]["true"] = []
                    sensor_map[u]["pred"] = []
                sensor_map[u]["true"].append((u+i, s))
                # if i[:-3] == "pm25-2017-05-01" and u in sens_test:
                #     print(u, i, round(float(s) * pm25_range/ 1000.0))
                test.append((u+i, s))
            line = f.readline()

    sens_test = ['173','189','196','201','222','228']
    # sens_test = [179, 184, 192, 203, 212, 221]
    sens_test = str(sens_test)
    test_avaliable = set([tup[0] for tup in test])
    with open("predict.txt", mode='r') as f:
        line = f.readline()
        while line:
            u, i, s = line.split()
            name = u+i
            # if i[:-3] == "pm25-2017-05-01" and u in sens_test:
            #     print(u, i, float(s) * pm25_range/ 1000.0)
            if str(u) in sens_test and name in test_avaliable:
                day = i
                day_map[day]["pred"].append((u+i, s))
                sensor_map[u]["pred"].append((u+i, s))
                pred.append((u+i, s))
            line = f.readline()

   
    # for tup in pred_temp:
    #     if tup[0] in test_avaliable:
    #         pred.append(tup)
    day_res_list = []
    max_day_r2 = -8.0
    max_day = ""
    min_day_r2 = 8.0
    for day in day_map:
        test_d = day_map[day]["true"]
        pred_d = day_map[day]["pred"]
        test_d =  sorted(test_d, key=lambda tup: tup[0], reverse=True)
        pred_d =  sorted(pred_d, key=lambda tup: tup[0], reverse=True)
        y_true = [float(tup[1]) * pm25_range/ 1000.0 for tup in test_d]
        y_pred= [max(0, float(tup[1]) * pm25_range/ 1000.0) for tup in pred_d]
        # x_label = [tup[0] for tup in test_d]
        # x_list = [i for i in range(len(test_d))]
        day_res_tup = (day, r2_score(y_true, y_pred))
        if day_res_tup[1] > max_day_r2:
            max_day = day
            max_day_r2 = day_res_tup[1]
        # max_day_r2 = max(max_day_r2, day_res_tup[1])
        min_day_r2 = min(min_day_r2, day_res_tup[1])
        # print(day_res_tup)

    print(max_day_r2, min_day_r2, max_day)
    sensor_res_list = []
    for sensor in sensor_map:
        test_s = sensor_map[sensor]["true"]
        pred_s = sensor_map[sensor]["pred"]
        test_s =  sorted(test_s, key=lambda tup: tup[0])
        pred_s =  sorted(pred_s, key=lambda tup: tup[0])
        y_true = [float(tup[1]) * pm25_range/ 1000.0 for tup in test_s]
        y_pred= [max(0, float(tup[1]) * pm25_range/ 1000.0) for tup in pred_s]
        sensor_res_tup = (sensor, r2_score(y_true, y_pred))
        sensor_res_list.append(sensor_res_tup)

    sensor_res_list = sorted(sensor_res_list, key=lambda tup: tup[1], reverse = True)
    print(sensor_res_list)

    test_h =  sorted(test, key=lambda tup: tup[0], reverse=True)
    pred_h =  sorted(pred, key=lambda tup: tup[0], reverse=True)
    y_true = [float(tup[1]) * pm25_range/ 1000.0 for tup in test_h]
    y_pred= [max(0, float(tup[1]) * pm25_range/ 1000.0) for tup in pred_h]
    print(r2_score(y_true, y_pred))

    

def clean_data():
    filenames = ["january-2017.csv","february-2017.csv","march-2017.csv","april-2017.csv","may-2017.csv","june-2017.csv","july-2017.csv","august-2017.csv","september-2017.csv","october-2017.csv","november-2017.csv","december-2017.csv"]
    #filenames = ["december-2017.csv"]
    path =r'./data/'
    frequency = 20  # this is the minimum sensor that you would expect to get. 
    # it would fail for some month if it can not find this mininum number of sensor that are avaliable at the same time.
    # 20 would usually works well.

    li = []
    fname_valid_data = {}
    f = open("valid_range.txt", "w")  # this is the output filename
    sens_test = ['172','173','189','196','201','222','228'] # the sensors that would not be included.

    for filename in filenames:
        fname = path + filename 
        df = pd.read_csv(fname, index_col=None, header=0)
        cols = [c for c in df.columns if c.lower()[:3].isdigit()]
        sensors = set([c.lower()[:3] for c in df.columns if c.lower()[:3].isdigit()])
        sensors = [sname for sname in sensors if sname not in sens_test] # candidate sensors
        names = ['temp', 'hum', 'pres', 'pm25'] # candidate features

        sensor_break_map = {} # records the valid range for each columns
        for sensor in sensors:
            sensor_break_map[sensor] = {}
            for name in names:
                sensor_break_map[sensor][name] = []

        cols = cols + ['UTC time']
        month_data = df[cols]
        for name in names:
            for cname in month_data.filter(regex=name):
                cur_sensor_id = cname.lower()[:3] 

                # for each sensor we want to find, find the continous part for each sensor
                if cur_sensor_id in sensors:
                    cur_col = month_data[cname]
                    is_nan = False
                    valid_range = (0, 0)
                    all_valid = True
                    for i in range(len(cur_col)):
                        if str(cur_col.iloc[i]).lower() == 'nan':
                            all_valid = False
                            if not is_nan:
                                valid_range = (valid_range[0], i-1)
                                if (valid_range[1] > valid_range[0]):
                                    sensor_break_map[cur_sensor_id][name].append(valid_range)
                            is_nan = True
                        else:
                            if is_nan:
                                valid_range = (i, valid_range[1])
                            is_nan = False
                    if all_valid: sensor_break_map[cur_sensor_id][name].append((0, len(cur_col) - 1))

        # for each sensor, obtain the range where all features are avaliable.
        clean_sensor_break_map = {}
        for sensor_id in sensor_break_map:
            expected_len = len(sensor_break_map[sensor_id]['temp'])
            for name in names:
                expected_len = min(len(sensor_break_map[sensor_id][name]), expected_len)
            clean_sensor_break_map[sensor_id] = []
            for i in range(expected_len):
                valid_range = sensor_break_map[sensor_id]['temp'][i]
                for name in names:
                    cur_range = sensor_break_map[sensor_id][name][i]
                    valid_range = (max(cur_range[0], valid_range[0]), min(cur_range[1], valid_range[1]))
                clean_sensor_break_map[sensor_id].append(valid_range)
        
        # try to find the maximum range for each sensor
        start_ranges = {}
        end_ranges = {}
        for sensor_id in clean_sensor_break_map:
            for element in clean_sensor_break_map[sensor_id]:
                s = element[0]
                e = element[1]
                if s in start_ranges:
                    start_ranges[s] += 1
                else:
                    start_ranges[s] = 1
                if e in end_ranges:
                    end_ranges[e] += 1
                else:
                    end_ranges[e] = 1
        start_ranges = [(e, start_ranges[e]) for e in start_ranges]
        end_ranges = [(e, end_ranges[e]) for e in end_ranges]
        start_ranges = sorted(start_ranges, key=lambda tup: tup[0])
        end_ranges = sorted(end_ranges, key=lambda tup: tup[0])
        
        # record the valid ranges where it contains the minimum number of sensors all avaliable
        start_pointer = -1
        end_pointer = -1
        cur_number = 0
        greater28_ranges = []
        while end_pointer < len(end_ranges) - 1:
            if cur_number >= frequency:
                greater28_ranges.append((start_ranges[start_pointer][0], end_ranges[end_pointer + 1][0]))
                end_pointer += 1
                cur_number -= end_ranges[end_pointer][1]
            else:
                if start_pointer == len(start_ranges) - 1: 
                    if cur_number >= frequency:
                        continue
                    else:
                        break
                start_pointer += 1
                cur_number += start_ranges[start_pointer][1]
                while end_pointer < len(end_ranges) - 1 and (end_ranges[end_pointer + 1][0] <= start_ranges[start_pointer][0]):
                    end_pointer += 1
                    cur_number -= end_ranges[end_pointer][1]
            
        # find the maximum number of rows
        max_range = (0, 0)
        max_len = 0
        for valid_range in greater28_ranges:
            cur_len = valid_range[1] - valid_range[0]
            if cur_len > max_len:
                max_len = cur_len
                max_range = valid_range
        if max_len == 0:
            print("IMPORTANT", fname) # if that data file do not have the number of minimum sensors that we expect, ERROR here

        final_max_sensors = [] 
        for sensor in clean_sensor_break_map:
            for v_range in clean_sensor_break_map[sensor]:
                if v_range[0] <= max_range[0] and v_range[1] >= max_range[1]:
                    final_max_sensors.append(sensor)
                    break
        # assert len(final_max_sensors) >= FREQENCY
        fname_valid_data[fname] = (max_range, final_max_sensors)

        # output to the file
        f.write(fname + ",")
        f.write(str(len(final_max_sensors)) + ",")
        f.write("(" + str(max_range[0]) + ",")
        f.write(str(max_range[1]) + "), ")
        final_max_sensors = sorted(final_max_sensors)
        for max_sensor in final_max_sensors:
            f.write(max_sensor + ",")
        f.write("\n")
    f.close()

if __name__ == "__main__":
    gen_file = False
    # for i in range(1,13):
    monthes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    pm25_range = gather_data(gen_file, monthes)
    if not gen_file:
        compare_data(pm25_range)
    # first_process_data("./data/sensor_locations.csv")
    # clean_data()
    

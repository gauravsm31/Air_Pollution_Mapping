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

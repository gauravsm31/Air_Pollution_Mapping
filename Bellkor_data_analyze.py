import numpy as np
import matplotlib.pyplot as plt

def avg_sensor():
    sensor_map = {}
    with open("test_sensor_trend.txt", 'r') as f:
        line = f.readline()
        while line:
            data = line.split("),")
            for element in data:
                sensor_id, accuracy = element.split(",")
                accuracy = accuracy[:-1]
                if sensor_id not in sensor_map:
                    sensor_map[sensor_id] = []
                sensor_map[sensor_id].append(float(accuracy))
            line = f.readline()
    end_list = []
    for sensor_id in sensor_map:
        sum_v = sum(sensor_map[sensor_id])
        end_list.append((sensor_id, sum_v * 1.0 / len(sensor_map[sensor_id])))
    end_list = sorted(end_list, key=lambda tup: tup[1], reverse=True)
    print(end_list)

def plot_monthly():
    fig, ax = plt.subplots()
    whole_list = [0.8691802174164558, 0.9259876580191129, 0.9079054926407887, 0.8714526851377938, 0.903071949603444, 
                  0.7211755283830575, 0.7822365690352129, 0.8221568425269487, 0.8953958527789494, 0.9274745653284036, 
                  0.915431342695249, 0.8887366709178587]
    monthly_list = [0.8999487404073844, 0.9167457631563553, 0.9039513803430783, 0.867284795556179, 0.9022309703569894, 
                    0.7173663574179246, 0.7856726081969495, 0.8198036625539763, 0.8929040943141983, 0.9263737132486596,
                    0.910548201735875, 0.8847241213583731]
    labels = ['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12']
    index = np.arange(len(labels))
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, whole_list, bar_width,alpha=opacity,color='b',label='A years data together')
    rects2 = plt.bar(index + bar_width, monthly_list, bar_width,alpha=opacity,color='g',label='A month data')

    plt.xlabel('Month')
    plt.ylabel('$R^2$ accuracy')
    plt.title("Different method vs. Accuracy")
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_05_01():
    pred_map = {}
    test_map = {}
    times = []
    with open("05_01_predict.txt", 'r') as f:
        line = f.readline()
        while line:
            u, i, r = line.split(',')
            if u not in pred_map:
                pred_map[u] = []
            pred_map[u].append((str(i), round(float(r))))
            times.append(i)
            line = f.readline()
    with open("05_01_real.txt", 'r') as f:
        line = f.readline()
        while line:
            u, i, r = line.split(',')
            if u not in test_map:
                test_map[u] = []
            test_map[u].append((str(i), round(float(r))))
            line = f.readline()
    test_sensors = [u for u in pred_map]
    times = set(times)
    times = sorted(times, key=lambda v:v)

    f1 = plt.figure(figsize=(8,4))
    
    for i in range(len(test_sensors)):
        ax1 = f1.add_subplot(2,3,1+i)
        index = np.arange(len(times))
        sensor_id = test_sensors[i]
        test_sensor_data = [v for v in test_map[sensor_id]]
        test_sensor_data = sorted(test_sensor_data, key=lambda tup:tup[0])
        test_sensor_data = [tup[1] for tup in test_sensor_data]

        pred_sensor_data = [v for v in pred_map[sensor_id]]
        pred_sensor_data = sorted(pred_sensor_data, key=lambda tup:tup[0])
        pred_sensor_data = [tup[1] for tup in pred_sensor_data]

        ax1.scatter(index,test_sensor_data, c='b', marker="o", label='Real')
        ax1.scatter(index,pred_sensor_data, c='r', marker="o", label='Prediction')

        # plt.xticks(index, times)
        ax1.set_xlabel('2017-05-01 hours')
        ax1.set_ylabel('Pollution Level (PM2.5)')
        # ax1.tick_params(axis='x')
        # ax1.tick_params(axis='y')
        ax1.set_title(sensor_id)

    f1.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # plot_monthly()
    # plot_05_01()
    avg_sensor()

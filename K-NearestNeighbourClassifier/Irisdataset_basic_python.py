import csv
import math
from collections import Counter

def find_euclidean_distance(formatted_data, test_data):
    distances_list = []
    for data in formatted_data:
        dat = data[0]
        dist = (abs(dat[i]-test_data[i])**2 for i in range(len(dat)))
        distances_list.append((math.sqrt(sum(dist)),data[1]))
    return distances_list


if __name__ == '__main__':
    #filter dataset
    formatted_data = []
    iris_csv = r"data\iris.csv"
    with open (iris_csv) as f:
        line_count = 0
        data = csv.reader(f, delimiter=',')
        for dat in data:
            if line_count == 0:
                line_count = 1
                continue
            formatted_data.append((list((map(float, dat[0:-1]))), dat[-1]))

    test_data = [3, 4, 5, 2]
    dist_list = find_euclidean_distance(formatted_data, test_data)

    nearest_neighbours = 5
    knn_list = sorted(dist_list, key=lambda x: x[0])[0:nearest_neighbours]
    #iris_type = max(knn_list,key=knn_list.count)
    iris_types = (i[-1] for i in knn_list)

    print("The data belongs to Iris: {}".format(max(iris_types)))



        

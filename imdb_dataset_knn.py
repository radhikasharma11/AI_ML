import csv
import math

def recommended_movies_list(dataset_path, test_data):
    relevant_data, result, raw_movie_data = [], [], []
    euclidean_dist = 0
    with open(dataset_path) as dataset:
        next(dataset)
        movie_data = dataset.readlines()
        for movie in movie_data:
            movie = movie.strip('\n').split(',')
            raw_movie_data.append(movie)
            movie = list(map(float, movie[2:]))
            relevant_data.append(movie)
    #print(relevant_data)

    for num, dat in enumerate(relevant_data):
        euclidean_dist = 0
        for i in range(len(dat)):
            euclidean_dist += (abs(dat[i]-test_data[i])**2)
        result.append((math.sqrt(euclidean_dist), num))
    result = sorted(result, key=lambda x: x[0])
    top_5_recommendations = result[:5]

    for n,t in top_5_recommendations:
        print(raw_movie_data[t][1])


if __name__ == '__main__':
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]  # feature vector of The Post
    dataset_path = r"C:\Users\Radhika\Desktop\datasets\movies_recommendation_data.csv"
    recommended_movies_list(dataset_path, the_post)

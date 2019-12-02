from scipy.io import loadmat

if __name__ == "__main__":
    dir = "models_lsvm\car_2_2012_10_04-20_13_57_model.mat"
    mat = loadmat(dir)
    # lon = mat['lon']
    # lat = mat['lat']
    # print(loadmat(dir)['lon'])
    print(mat)
import pickle

def save_pickle_data(save_data_dir, save_data, dataname):
    # 保存数据集
    with open(save_data_dir + "/" + dataname + ".pickle", "wb") as handle:
        # pickle.dump(save_data, handle)
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

def read_pickle_data(save_data_dir, dataname):
    with open(save_data_dir + "/" + dataname + ".pickle", 'rb') as handle:
        read_data = pickle.load(handle)
    handle.close()
    return read_data
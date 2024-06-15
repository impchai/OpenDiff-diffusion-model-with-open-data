import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def parse_id():

    observed_values = np.load('./data_shanghai.npz')['bs_record']
    min_val = np.min(observed_values, axis=-1)
    max_val = np.max(observed_values, axis=-1)
    normalized_data = (observed_values.T- min_val) / (max_val - min_val)
    observed_values = normalized_data.T


    return observed_values




class Traffic_Dataset(Dataset):
    def __init__(self, eval_length=168, use_index_list=None,seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        path = (
            "./data/traffic_volumn" + "_seed" + str(seed) + ".pk"
        )


        self.observed_values = np.array(parse_id())
        with open(path, "wb") as f:
            pickle.dump(
                self.observed_values, f
            )
        with open(path, "rb") as f:
            self.observed_values = pickle.load(
                f
            )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

        with open('region_embeding_sh.txt', 'r') as file:
            # 读取所有内容
            content = file.read()
        number_list = content.split()
        # 然后，将列表转换为NumPy数组
        self.region_embed = np.array(number_list, dtype=float).reshape(-1, 128)  # 使用适当的数据类型

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "region_emb": self.region_embed[index],
            "timepoints": np.arange(self.eval_length),
            "idex_test": index,
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16):

    dataset = Traffic_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    


#------------------------------
    train_size = 0.6
    valid_size = 0.2  # 验证集大小
    test_size = 0.2  # 测试集大小
    train_end = int(len(indlist) * train_size)
    valid_end = train_end + int(len(indlist) * valid_size)

    # 随机打乱索引
    np.random.seed(seed)  # 设置随机种子以保持结果的一致性
    np.random.shuffle(indlist)

    train_index = indlist[:train_end]
    valid_index = indlist[train_end:valid_end]
    test_index = indlist[valid_end:]

    train_dataset = Traffic_Dataset(use_index_list=train_index, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Traffic_Dataset(use_index_list=valid_index, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Traffic_Dataset(use_index_list=test_index, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    return train_loader, valid_loader, test_loader




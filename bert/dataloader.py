import random

class MyDatasetBase():
    def __init__(self) -> None:
        pass
    def __call__(self,index):
        yield self.__getitem__(index)
    def __getitem__(self):
        pass
    def __len__(self):
        pass

class MyDataset(MyDatasetBase):
    def __init__(self, *myliststuple) -> None:
        super().__init__()
        self.myliststuple = myliststuple

    def __getitem__(self, index):
        elelist = []
        for mylist in self.myliststuple:
            elelist.append(mylist[index])
        return elelist

    def __len__(self):
        return len(self.myliststuple[0])

class MyDataloader():
    def __init__(self, dataset, batch_size, shuffle=True) -> None:
        self.dataset = dataset
        self.batchsize = batch_size
        self.batchcnt = int((len(self.dataset) + batch_size - 1) / batch_size)
        self.len = self.__len__()
        if shuffle:
            self.index = [i for i in range(self.len)]
            random.shuffle(self.index)

    def __iter__(self):
        for i in range(self.batchcnt):
            start = self.batchsize * i
            end = min(start + self.batchsize, self.len)
            itetuple = ()
            for mylist in self.dataset.myliststuple:
                mylist_iter_shuffled = [mylist[i] for i in self.index[start:end]]
                itetuple = itetuple.__add__((mylist_iter_shuffled,))
            yield itetuple

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    mylist1, mylist2, mylist3 = [1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 20, 30, 40, 50, 60, 70, 80, 90], [100, 200, 300, 400,
                                                                                                    500, 600, 700, 800,
                                                                                                    900]
    T = MyDataset(mylist1, mylist2, mylist3)
    t_iter = MyDataloader(T, 2)
    for i in t_iter:
        print(i)
    t_iter = MyDataloader(T, 5)
    for i in t_iter:
        print(i)
    t_iter = MyDataloader(T, 3)
    for i in t_iter:
        print(i)

"""    
([9, 8], [90, 80], [900, 800])
([7, 5], [70, 50], [700, 500])
([2, 6], [20, 60], [200, 600])
([4, 1], [40, 10], [400, 100])
([3], [30], [300])
([8, 2, 7, 1, 4], [80, 20, 70, 10, 40], [800, 200, 700, 100, 400])
([6, 5, 9, 3], [60, 50, 90, 30], [600, 500, 900, 300])
([3, 2, 9], [30, 20, 90], [300, 200, 900])
([1, 7, 5], [10, 70, 50], [100, 700, 500])
([4, 6, 8], [40, 60, 80], [400, 600, 800])

([6, 5], [60, 50], [600, 500])
([3, 1], [30, 10], [300, 100])
([7, 9], [70, 90], [700, 900])
([2, 8], [20, 80], [200, 800])
([4], [40], [400])
([9, 7, 3, 2, 1], [90, 70, 30, 20, 10], [900, 700, 300, 200, 100])
([8, 5, 6, 4], [80, 50, 60, 40], [800, 500, 600, 400])
([3, 2, 1], [30, 20, 10], [300, 200, 100])
([6, 8, 9], [60, 80, 90], [600, 800, 900])
([7, 4, 5], [70, 40, 50], [700, 400, 500])
"""
# 日期：2021年07月17日
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as tsf
import cv2


batch_size = 64
transform = tsf.Compose([tsf.ToTensor(), tsf.Normalize([0.1307], [0.3081])])
# Normalize:正则化，降低模型复杂度,防止过拟合

# 下载数据集
# torchvision已经预先实现了常用的Dataset，包括MINST。可以用datasets.MNIST直接从网上下载，并自动建立名为data的文件夹。
train_set=datasets.MNIST(root="data",train=True,download=True,transform=transform)
test_set=datasets.MNIST(root="data",train=False,download=True,transform=transform)

# 加载数据集，将数据集变成迭代器
def get_data_loader():
    train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
    return train_loader,test_loader


# 显示数据集中的图片
# with open("data/MNIST/raw/train-images-idx3-ubyte","rb") as f:
#     file=f.read()
#     image1=[int(str(item).encode('ascii'),16) for item in file[16:16+784]]
#     image1_np=np.array(image1,dtype=np.uint8).reshape(28,28,1)
#     cv2.imshow("image1_np",image1_np)
#     cv2.waitKey(0)










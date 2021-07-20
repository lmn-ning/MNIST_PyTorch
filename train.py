# 日期：2021年07月17日
import torch
import cnn
import torch.nn.functional as F
from dataset import get_data_loader
import torch.optim as optim


if __name__ == "__main__":

    # 超参
    batch_size=64
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 5

    # 选择模型
    model=cnn.CNN().to(device)

    # 定义优化器
    optimizer=optim.Adam(model.parameters())

    # 加载迭代器
    train_loader, test_loader = get_data_loader()

    # 训练
    def train(epoch_i):
        model.train()   # 设置为训练模式
        for batch_i,(digit,label) in enumerate(train_loader):
            digit,label=digit.to(device),label.to(device)
            optimizer.zero_grad()    # 梯度初始化为0
            output=model(digit)     # 训练结果,output是概率
            loss=F.cross_entropy(output,label)     # 定义损失函数,交叉熵损失函数适用于多分类问题
            loss.backward()    # 反向传播
            optimizer.step()    # 更新参数

            if batch_i % 100 == 0:
                print("train    epoch_i: {}    batch_i: {}    loss: {: .8f}".format(epoch_i,batch_i,loss.item()))

    # 测试
    def test(epoch_i):
        model.eval()    # 设置为测试模式
        acc = 0.
        loss = 0.
        with torch.no_grad():
            for digit, label in test_loader:
                digit, lable = digit.to(device), label.to(device)
                output = model(digit)  # 模型输出
                loss += F.cross_entropy(output, lable).item()

                predict = output.max(dim=1, keepdim=True)[1]
                # 找到概率最大值的下标，1表示按行计算。
                # max()返回两个值，第一个是值，第二个是索引，所以取 max[1]

                acc += predict.eq(label.view_as(predict)).sum().item()
            accuracy = acc / len(test_loader.dataset) * 100
            test_loss = loss / len(test_loader.dataset)
            print("test     epoch_i: {}    loss: {: .8f}    accuracy: {: .4f}%".format(epoch_i,test_loss,accuracy))

    # train && test
    for epoch_i in range(1,epoch+1):
        train(epoch_i)
        test(epoch_i)

    # 保存模型
    torch.save(model,"save_model/model.pt")







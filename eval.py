# 日期：2021年07月17日
import torch
from dataset import get_data_loader


if __name__ == "__main__":
    _, eval_loader = get_data_loader()   # 因为没有验证集，所以将测试题作为验证集使用。
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("save_model/model.pt")    # 加载模型
    # model.eval()  # 设置为验证模式

    acc = 0.
    with torch.no_grad():
        for digit, label in eval_loader:
            digit, lable = digit.to(device), label.to(device)
            output = model(digit)  # 模型输出
            predict = output.max(dim=1, keepdim=True)[1]
            # 找到概率最大值的下标，1表示按行计算。
            # max()返回两个值，第一个是值，第二个是索引，所以取 max[1]

            acc += predict.eq(label.view_as(predict)).sum().item()
        acceracy = acc/len(eval_loader.dataset) * 100
        print("eval accuracy: {: .4f}%".format(acceracy))
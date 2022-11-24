import time

from modules.BasicBlock import *

class STSCNN(nn.Module):
    def __init__(self):
        super(STSCNN, self).__init__()
        self.conv_1 = BasicConv(1, 30, kernel_size=3, relu=False, stride=1)
        self.conv_2 = BasicConv(1, 30, kernel_size=3, relu=False, stride=1)
        self.conv_3 = BasicConv(1, 60, kernel_size=3, relu=False, stride=1)
        self.relu_1_2 = nn.ReLU(inplace=True)
        self.feature3 = BasicConv(60, 20, kernel_size=3, relu=False, stride=1)
        self.feature5 = BasicConv(60, 20, kernel_size=5, relu=False, stride=1)
        self.feature7 = BasicConv(60, 20, kernel_size=7, relu=False, stride=1)
        self.relu_feature = nn.ReLU(inplace=True)
        self.conv1 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1)

        self.conv2 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1)
        # conv_3 + relu2
        self.conv3 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1, dilat=2, padding=2)
        self.conv4 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1, dilat=3, padding=3)
        self.conv5 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1, dilat=2, padding=2)

        self.conv6 = BasicConv(60, 60, kernel_size=3, relu=True, stride=1)

        self.conv7 = BasicConv(60, 1, kernel_size=3, relu=False, stride=1)


    def forward(self, data1, data2, mask):
        # data1 有云， data2清晰， data3 融合图像
        data3 = data1 + data2 * (1 - mask)
        y1 = self.conv_1(data1)
        y2 = self.conv_2(data2)
        y3 = self.conv_3(data3)
        y = torch.cat([y1, y2], dim=1)
        feature1 = self.feature3(y)
        feature2 = self.feature5(y)
        feature3 = self.feature7(y)

        feature = torch.cat([feature1, feature2, feature3], dim=1)
        feature = self.relu_feature(feature + y)
        y4 = self.conv1(feature)
        x = self.conv2(y4)
        x = x + y3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x + y3
        x = y4 + self.conv6(x)
        out = self.conv7(x) + data1
        return out

if __name__ == '__main__':
    model = STSCNN()

    image1 = torch.randn(6, 1, 256, 256)
    image2 = torch.randn(6, 1, 256, 256)
    mask = torch.randn(6, 1, 256, 256)

    with torch.no_grad():
        output1 = model.forward(image1, image2, mask)
    print(output1.size())


import torch.nn as nn 
import torch.nn.functional as F
import torch


def add_vgg():
    layers = []


    # stage 1
    con1_1 = nn.Conv2d(3,64,3,1,1)
    bn1_1 = nn.BatchNorm2d(64)
    relu_1_1 = nn.ReLU(inplace=True)
    con1_2 = nn.Conv2d(64,64,3,1,1)
    bn1_2 = nn.BatchNorm2d(64)
    relu_1_2 = nn.ReLU(inplace=True) 
    maxpool_1 = nn.MaxPool2d(2,2,0,1) # bash_size,64,150,150

    layers.append(con1_1)
    layers.append(bn1_1)
    layers.append(relu_1_1)
    layers.append(con1_2)
    layers.append(bn1_2)
    layers.append(relu_1_2)
    layers.append(maxpool_1)
    


    # stage 2
    con2_1 = nn.Conv2d(64,128,3,1,1)
    bn2_1 = nn.BatchNorm2d(128)
    relu_2_1 = nn.ReLU(inplace=True) 
    con2_2 = nn.Conv2d(128,128,3,1,1)
    bn2_2 = nn.BatchNorm2d(128)
    relu_2_2 = nn.ReLU(inplace=True) 
    maxpool_2 = nn.MaxPool2d(2,2,0,1) # bash_size,128,75,75

    layers.append(con2_1)
    layers.append(bn2_1)
    layers.append(relu_2_1)
    layers.append(con2_2)
    layers.append(bn2_2)
    layers.append(relu_2_2)
    layers.append(maxpool_2)

    #stage 3
    con3_1 = nn.Conv2d(128,256,3,1,1)
    bn3_1 = nn.BatchNorm2d(256)
    relu_3_1 = nn.ReLU(inplace=True) 
    con3_2 = nn.Conv2d(256,256,3,1,1)
    bn3_2 = nn.BatchNorm2d(256)
    relu_3_2 = nn.ReLU(inplace=True)
    con3_3 = nn.Conv2d(256,256,3,1,1)
    bn3_3 = nn.BatchNorm2d(256)
    relu_3_3 = nn.ReLU(inplace=True)
    maxpool_3 = nn.MaxPool2d(2,2,0,1,ceil_mode=True) # bash_size,256,38,38

    layers.append(con3_1)
    layers.append(bn3_1)
    layers.append(relu_3_1)
    layers.append(con3_2)
    layers.append(bn3_2)
    layers.append(relu_3_2)
    layers.append(con3_3)
    layers.append(bn3_3)
    layers.append(relu_3_3)
    layers.append(maxpool_3)


    #stage 4
    con4_1 = nn.Conv2d(256,512,3,1,1)
    bn4_1 = nn.BatchNorm2d(512)
    relu_4_1 = nn.ReLU(inplace=True) 
    con4_2 = nn.Conv2d(512,512,3,1,1)
    bn4_2 = nn.BatchNorm2d(512)
    relu_4_2 = nn.ReLU(inplace=True)
    con4_3 = nn.Conv2d(512,512,3,1,1) # ---------->
    bn4_3 = nn.BatchNorm2d(512)
    relu_4_3 = nn.ReLU(inplace=True)
    maxpool_4 = nn.MaxPool2d(2,2,0,1) # bash_size,512,19,19

    layers.append(con4_1)
    layers.append(bn4_1)
    layers.append(relu_4_1)
    layers.append(con4_2)
    layers.append(bn4_2)
    layers.append(relu_4_2)
    layers.append(con4_3)
    layers.append(bn4_3)
    layers.append(relu_4_3)
    layers.append(maxpool_4)

    #stage 5
    con5_1 = nn.Conv2d(512,512,3,1,1)
    bn5_1 = nn.BatchNorm2d(512)
    relu_5_1 = nn.ReLU(inplace=True) 
    con5_2 = nn.Conv2d(512,512,3,1,1)
    bn5_2 = nn.BatchNorm2d(512)
    relu_5_2 = nn.ReLU(inplace=True)
    con5_3 = nn.Conv2d(512,512,3,1,1)
    bn5_3 = nn.BatchNorm2d(512)
    relu_5_3 = nn.ReLU(inplace=True)
    maxpool_5 = nn.MaxPool2d(3,1,1,1) # bash_size,512,19,19

    layers.append(con5_1)
    layers.append(bn5_1)
    layers.append(relu_5_1)
    layers.append(con5_2)
    layers.append(bn5_2)
    layers.append(relu_5_2)
    layers.append(con5_3)
    layers.append(bn5_3)
    layers.append(relu_5_3)
    layers.append(maxpool_5)

    # stage 6
    conv6 = nn.Conv2d(512,1024,3,1,6,6) # bash_size,1024,19,19
    relu6 = nn.ReLU(inplace=True)

    layers.append(conv6)
    layers.append(relu6)

    # stage 7 
    conv7 = nn.Conv2d(1024,1024,1) # ----------> # bash_size,1024,19,19
    relu7 = nn.ReLU(inplace=True)

    layers.append(conv7)
    layers.append(relu7)

    return layers


def add_extras(i, size):
    layers = []

    # stage 8 
    conv8_1 = nn.Conv2d(i,256,1,1) # bash_size,256,19,19
    conv8_2 = nn.Conv2d(256,512,3,2,1) # bash_size,512,10,10 # ---------->

    layers.append(conv8_1)
    layers.append(conv8_2)

    # stage 9
    conv9_1 = nn.Conv2d(512,128,1,1) # bash_size,128,19,19
    conv9_2 = nn.Conv2d(128,256,3,2,1) # bash_size,256,5,5 # ---------->

    layers.append(conv9_1)
    layers.append(conv9_2)

    # stage 10
    conv10_1 = nn.Conv2d(256,128,1,1) # bash_size,128,19,19
    conv10_2 = nn.Conv2d(128,256,3,2,1) # bash_size,256,3,3 # ---------->

    layers.append(conv10_1)
    layers.append(conv10_2)

    # stage 11
    conv11_1 = nn.Conv2d(256,128,1,1) # bash_size,128,19,19
    conv11_2 = nn.Conv2d(128,256,3,1) # bash_size,256,1,1 # ---------->

    layers.append(conv11_1)
    layers.append(conv11_2)

    return layers



class vgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #input_size = cfg.INPUT.IMAGE_SIZE # 输入图片大小 300
        input_size = 300

        self.vgg = nn.ModuleList(add_vgg())
        self.extras = nn.ModuleList(add_extras(i=1024,size=input_size))

        self.reset_parameters() # 初始化 extras 的权重

    # 初始化 extras 的权重
    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # 初始化 vgg部分的权重
    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    
    def forward(self, x):
        features = []

        for i in range(31):
            x = self.vgg[i](x)
        # todo   l2nom
        features.append(x)  # bash_size,256,38,38

        for i in range(31,len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x) # bash_size,256,19,19

        for k , v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x) # 1,3,5  # 10,5,3,1

        return tuple(features)


if __name__ == "__main__":
    vgg = vgg(1)

    data = torch.rand(12,3,300,300)

    result = vgg(data)
    for i in result:
        print(i.size())
import torch
from torch import nn
import torch.nn.functional as F


class ICVAE(nn.Module):
    def __init__(self, cfg, device):
        super(ICVAE, self).__init__()
        self.device = device

        self.cls_loss_coef = cfg.cls_loss_coef

        self.num_instances = cfg.num_instances
        self.num_classes = cfg.num_classes
        self.num_memories = cfg.num_memories

        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.addressing = cfg.addressing
        self.conv_channel_size = cfg.conv_channel_size

        self.feature_size = self.conv_channel_size*4 * 4 * 4
        self.drop_rate = cfg.drop_rate

        self.encoder = Encoder(image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                              )

        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)

        self.cosine_similarity = nn.CosineSimilarity(dim=2,)

        self.decoder = Decoder(image_height=self.image_height,
                               image_width=self.image_width,
                               image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                              )

        self.relu = nn.ReLU(inplace=True)

        if self.cls_loss_coef > 0.0:
            self.classifier = Classifier(image_channel_size=self.image_channel_size,
                                         conv_channel_size=self.conv_channel_size,
                                         num_classes=self.num_classes,
                                         drop_rate=self.drop_rate,
                                        )

        if self.addressing == 'sparse':
            self.threshold = 1 / self.memory.size(0)
            self.epsilon = 1e-15

    def forward(self, x):
        batch, channel, height, width = x.size()

        z = self.encoder(x)

        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)

        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        if self.addressing == 'soft':
            z_hat = torch.mm(mem_weight, self.memory)
        elif self.addressing == 'sparse':
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                            / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1) \
                            .unsqueeze(1).expand(batch, self.num_memories)
            z_hat = torch.mm(mem_weight, self.memory)

        rec_x = self.decoder(z_hat)

        if self.cls_loss_coef > 0.0:
            logit_x = self.classifier(rec_x)
        else:
            logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        return dict(rec_x=rec_x, logit_x=logit_x, mem_weight=mem_weight)

    def generate_from_memory(self, idx):
        z_hat = self.memory[idx]
        batch, _ = z_hat.size()

        rec_x = self.decoder(z_hat)

        if self.cls_loss_coef > 0.0:
            logit_x = self.classifier(rec_x)
        else:
            logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        return dict(rec_x=rec_x, logit_x=logit_x)


class Encoder(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
                               out_channels=self.conv_channel_size,
                               kernel_size=1,
                               stride=2,
                               padding=1,
                              )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size,
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size*2,
                               out_channels=self.conv_channel_size*4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                              )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        batch, _, _, _ = x.size()
        x = x.view(batch, -1)
        return x


class Decoder(nn.Module):
    def __init__(self, image_height, image_width, image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
                                          out_channels=self.conv_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.image_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                         )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.conv_channel_size*4, 4, 4)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size, num_classes, drop_rate):
        super(Classifier, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
                               out_channels=6,
                               kernel_size=5,
                              )

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                              )

        self.fc1 = nn.Linear(in_features=16*4*4, out_features=256,)
        self.fc2 = nn.Linear(in_features=256, out_features=128,)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes,)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2,)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        return x

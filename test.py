from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import BatchCollator


class Tester():
    def __init__(self, cfg, dataloader, model, device):
        self.cfg = cfg
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.image_channel_size = cfg.image_channel_size

        self.num_dataloaders = cfg.num_dataloaders
        self.device = device

        self.batch_size = cfg.batch_size

        self.model = model

        self.cls_loss_coef = cfg.cls_loss_coef
        self.entropy_loss_coef = cfg.entropy_loss_coef
        self.condi_loss_coef = cfg.condi_loss_coef
        self.addressing = cfg.addressing
        self.num_memories = cfg.num_memories

        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.rec_criterion = nn.MSELoss(reduction='sum')
        self.condi_criterion = nn.BCELoss(reduction='sum')

        if cfg.test_set == 'train':
            self.test_set = dataloader.train_dataset
        else:
            self.test_set = dataloader.test_dataset

        self.collator = BatchCollator(image_height=self.image_height,
                                      image_width=self.image_width,
                                      image_channel_size=self.image_channel_size,
                                     )

    def test(self):
        records = dict(loss=[],
                       rec_loss=[],
                       entropy_loss=[],
                       condi_loss=[],
                       rec_error=[],
                       cls_loss=[],
                       cls_acc=[],
                      )

        self.testloader = DataLoader(dataset=self.test_set,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     collate_fn=self.collator,
                                     num_workers=self.num_dataloaders,
                                    )
        for i, batch in tqdm(enumerate(self.testloader), total=len(self.testloader), desc='Test'):
            self.model.eval()

            batch = [b.to(self.device) for b in batch]
            imgs, labels, instances = batch[0], batch[1], batch[2]
            batch_size = imgs.size(0)

            with torch.no_grad():
                result = self.model(imgs)

            rec_imgs = result['rec_x']
            cls_logit = result['logit_x']
            mem_weight = result['mem_weight']

            if self.condi_loss_coef > 0.0:
                onehot_c = torch.FloatTensor(batch_size, self.num_memories).to(self.device)
                onehot_c.zero_()
                onehot_c.scatter_(1, instances.unsqueeze(1), 1.0)
                condi_loss = self.condi_criterion(mem_weight, onehot_c)
                condi_loss /= batch_size
                condi_loss *= self.condi_loss_coef
            else:
                condi_loss = torch.zeros(1).to(self.device)

            if self.addressing == 'sparse':
                mask = (mem_weight == 0).float()
                maksed_weight = mem_weight + mask
                entropy_loss = -mem_weight * torch.log(maksed_weight)
                entropy_loss = entropy_loss.sum() / batch_size
                entropy_loss *= self.entropy_loss_coef
            else:
                entropy_loss = torch.zeros(1).to(self.device)

            rec_loss = self.rec_criterion(rec_imgs, imgs)
            rec_loss /= batch_size
            rec_error = (rec_imgs - imgs).pow(2).sum(1).sum(1).sum(1)

            if self.cls_loss_coef > 0.0:
                cls_loss = self.cls_criterion(cls_logit, instances)
                cls_loss *= self.cls_loss_coef
            else:
                cls_loss = torch.zeros(1).to(self.device)
            cls_pred = cls_logit.max(1)[1]
            cls_acc = (cls_logit.max(1)[1] == instances).float()

            loss = rec_loss + cls_loss + entropy_loss + condi_loss

            records['loss'] += [loss.cpu().item()]
            records['rec_loss'] += [rec_loss.cpu().item()]
            records['entropy_loss'] += [entropy_loss.cpu().item()]
            records['condi_loss'] += [condi_loss.cpu().item()]
            records['rec_error'] += rec_error.cpu().tolist()
            records['cls_loss'] += [cls_loss.cpu().item()]
            records['cls_acc'] += cls_acc.cpu().tolist()

        for k, v in records.items():
            records[k] = sum(records[k]) / len(records[k])

        loss = records['loss']
        rec_loss = records['rec_loss']
        rec_error = records['rec_error']
        entropy_loss = records['entropy_loss']
        condi_loss = records['condi_loss']
        cls_loss = records['cls_loss']
        cls_acc = records['cls_acc']

        print('='*100)
        print('Test')
        print('Reconst error: {rec_error:.4f}'.format(rec_error=rec_error, end=' '))
        print('Loss: {loss:.4f}, Reconst loss: {rec_loss:.4f}' \
                  .format(loss=loss, rec_loss=rec_loss, end=' '))
        print('Entropy loss: {entropy_loss:8f}' \
                  .format(entropy_loss=entropy_loss, end=' '))
        print('Condition loss: {condi_loss:4f}' \
                  .format(condi_loss=condi_loss, end=' '))
        print('Cls_loss: {cls_loss:.4f}, Cls acc: {cls_acc:.4f}' \
                  .format(cls_loss=cls_loss, cls_acc=cls_acc, end=' '))
        print()

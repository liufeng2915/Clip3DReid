import time
import torch
from tools.utils import AverageMeter

def train(epoch, model, clip_model, optimizer, trainloader):

    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, clip_feat, betas, pids, camids, _) in enumerate(trainloader):

        imgs, pids  = imgs.cuda(), pids.cuda()
        text_feat = clip_feat[:,1:,:].cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        loss, preds = model.forward(clip_model, text_feat, imgs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'Acc:{acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time,
                   acc=corrects))


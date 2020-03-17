import torch.nn as nn
import time
import copy
import torch
from tqdm import tqdm
import os
import numpy as np

def train_model(path, writer, model, dataloaders, criterion, optimizer, device, num_epochs=5):
    sigmoid = nn.Sigmoid()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0.0

    for epoch in range(num_epochs):

        print('-' * 10)

        print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, num_epochs, best_acc))

        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            if phase == 'val':
                model.eval()
            running_loss = 0.0
            nbre_sample = 0
            TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
            train_problems, train_filename = dataloaders[phase].get_next()
            train_bar = tqdm(train_problems)
            for index_pb, problem in enumerate(train_bar):
                n_batches = len(problem.is_sat)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(problem)
                    target = torch.Tensor(problem.is_sat).float().to(model.L_init.weight.device)
                    outputs = sigmoid(outputs)
                    loss = criterion(outputs, target)
                    desc = 'loss: %.4f; ' % (loss.item())
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.65)
                        optimizer.step()
                    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device))
                    TP += (preds.eq(1) & target.eq(1)).cpu().sum()
                    TN += (preds.eq(0) & target.eq(0)).cpu().sum()
                    FN += (preds.eq(0) & target.eq(1)).cpu().sum()
                    FP += (preds.eq(1) & target.eq(0)).cpu().sum()
                    TOT = TP + TN + FN + FP
                    desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                    (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(),
                    TN.item() * 1.0 / TOT.item(), FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
                    running_loss += loss.item() * n_batches
                    nbre_sample += n_batches

            epoch_loss = running_loss / nbre_sample
            acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            print('{} Acc: {:.4f}'.format(
                phase, acc))
            print(desc)
            writer.add_scalar(phase + ' loss',
                              epoch_loss,
                              epoch)
            writer.add_scalar(phase + ' Acc',
                              acc,
                              epoch)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch + 1, 'acc': best_loss, 'state_dict': model.state_dict()},
                           os.path.join(path, str(best_loss) + '_bestloss.pth.tar'))
            if phase == 'val' and acc >= best_acc:
                best_acc = acc
                torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': model.state_dict()},
                           os.path.join(path, str(best_acc) + '_bestacc.pth.tar'))
        print()
    torch.save({'epoch': epoch + 1, 'acc': acc, 'state_dict': model.state_dict()},
               os.path.join(path, 'model_last.pth.tar'))

    print()
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("---" * 100)
    print()
    print()
    print("Start test")

    problems_test, test_filename = dataloaders["test"].get_next()
    solve_pb(problems_test, model, path)



    # load best model weights
    model.load_state_dict(best_model_wts)
    #TODO : save
    return model

def solve_pb(problems_test, model, path):
    test_bar = tqdm(problems_test)
    compteur = 0.0
    total = 0.0
    sigmoid = nn.Sigmoid()
    device = model.L_init.weight.device
    TP, TN, FN, FP = 0, 0, 0, 0
    times = []

    for _, problem in enumerate(test_bar):
        start_time = time.time()
        outputs = model(problem)
        outputs = sigmoid(outputs)
        preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).to(device), torch.zeros(outputs.shape).to(device)).cpu().detach().numpy()

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        times.append(duration)

        target = np.array(problem.is_sat)
        TP += int(((preds == 1) & (target == 1)).sum())
        TN += int(((preds == 0) & (target == 0)).sum())
        FN += int(((preds == 0) & (target == 1)).sum())
        FP += int(((preds == 1) & (target == 0)).sum())

        num_cases = TP + TN + FN + FP

    print(sum(times), len(times), sum(times) * 1.0 / len(times))
    print()
    print(num_cases, TP,  TN,  FN,  FP)
    print()
    print((TP + TN) * 1.0 / num_cases, 2 * TP * 1.0 / num_cases, 2 * TN * 1.0 / num_cases, 2 * FN * 1.0 / num_cases,
                  2 * FP * 1.0 / num_cases)

    for _, problem in enumerate(test_bar):
        solutions, fl, fc = model.find_solutions(problem, model, path, flag_plot= True)
        target = np.array(problem.is_sat)
        for batch, solution in enumerate(solutions):
            if target[batch]:
                total += 1
                if solution is not None:
                    # print("[%s] %s" % (problem.dimacs[batch], str(solution)))
                    compteur += 1

    print('Pourcentage pb solved variables: {:4f} %'.format(100 * compteur / total))
    print('number sat solved: ', (compteur))
    print('number total sat to solve: ', (total))


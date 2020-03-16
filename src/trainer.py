import torch.nn as nn
import time
import copy
import torch
from tqdm import tqdm
import os


def train_model(path, writer, model, dataloaders, criterion, optimizer,device, num_epochs=5):
    sigmoid  = nn.Sigmoid()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, num_epochs, best_acc))




        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            nbre_sample = 0
            TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()

            #while dataloaders[phase].has_next():
            # Iterate over data.
            train_problems, train_filename = dataloaders[phase].get_next()

            train_bar = tqdm(train_problems)

            for index_pb, problem in enumerate(train_bar):

                n_batches = len(problem.is_sat)



                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(problem)
                    target = torch.Tensor(problem.is_sat).float()
                    # print(outputs.shape, target.shape)
                    # print(outputs, target)
                    outputs = sigmoid(outputs)
                    loss = criterion(outputs, target)
                    desc = 'loss: %.4f; ' % (loss.item())

                    if phase == 'train':

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.65)
                        optimizer.step()


                    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape), torch.zeros(outputs.shape))

                    TP += (preds.eq(1) & target.eq(1)).cpu().sum()
                    TN += (preds.eq(0) & target.eq(0)).cpu().sum()
                    FN += (preds.eq(0) & target.eq(1)).cpu().sum()
                    FP += (preds.eq(1) & target.eq(0)).cpu().sum()
                    TOT = TP + TN + FN + FP

                    desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % (
                    (TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(),
                    TN.item() * 1.0 / TOT.item(), FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
                    # train_bar.set_description(desc)



                    # backward + optimize only if in training phase



                    # statistics
                    running_loss += loss.item() * n_batches


                    #running_corrects += torch.sum(preds == labels.data)
                    nbre_sample += n_batches

            if (index_pb%100):


                epoch_loss = running_loss / nbre_sample
                acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
                #epoch_acc = running_corrects.double() / nbre_sample

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))




                print('{} Acc: {:.4f}'.format(
                    phase, acc))

                print(desc)


            epoch_loss = running_loss / nbre_sample
            acc = (TP.item() + TN.item()) * 1.0 / TOT.item()
            # epoch_acc = running_corrects.double() / nbre_sample

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


            if phase == 'val' and acc >= best_acc:
                best_acc = acc

            torch.save({'epoch': epoch + 1, 'acc': acc, 'state_dict': model.state_dict()},
                       os.path.join(path, str(epoch_loss) + '_last.pth.tar'))
            if phase == 'val' and acc >= best_acc:
                best_acc = acc
                torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': model.state_dict()},
                           os.path.join(path, str(best_acc) + '_best.pth.tar'))


        print()

    problems_test, train_filename = dataloaders["test"].get_next()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))


    solve_pb(problems_test, model, path)



    # load best model weights
    model.load_state_dict(best_model_wts)
    #TODO : save
    return model

def solve_pb(problems_test, g, path):
    test_bar = tqdm(problems_test)
    compteur = 0.0
    total = 0.0
    for _, problem in enumerate(test_bar):
        solutions = g.find_solutions(problem, g, path)
        for batch, solution in enumerate(solutions):
            total += 1
            if solution is not None:
                print("[%s] %s" % (problem.dimacs[batch], str(solution)))
                compteur +=1

    print('Pourcentage pb solved variables: {:4f}'.format(compteur/total))


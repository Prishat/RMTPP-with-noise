from util import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def evaluate():
    model.eval()
    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        gold_times.append(batch[0][:, -1].numpy())
        gold_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch)
        pred_times.append(pred_time)
        pred_events.append(pred_event)
    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    pred_events = np.concatenate(pred_events).reshape(-1)
    gold_events = np.concatenate(gold_events).reshape(-1)
    time_error = abs_error(pred_times, gold_times)
    acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
    print(f"epoch {epc}")
    print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")


def evaluate_noise():
    model.eval()

    errors = np.linspace(0,10,200)
    abs_err = []

    for z in errors:
        pred_times, pred_events = [], []
        gold_times, gold_events = [], []
        for i, batch in enumerate(tqdm(test_loader)):
            batch[0][0] += z
            gold_times.append(batch[0][:, -1].numpy())
            gold_events.append(batch[1][:, -1].numpy())
            pred_time, pred_event = model.predict(batch)
            #loss = model(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        pred_times = np.concatenate(pred_times).reshape(-1)
        gold_times = np.concatenate(gold_times).reshape(-1)
        pred_events = np.concatenate(pred_events).reshape(-1)
        gold_events = np.concatenate(gold_events).reshape(-1)
        time_error = abs_error(pred_times, gold_times)
        abs_err.append(time_error)
        print("time_error := ", time_error)
        #acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
        #print(f"epoch {epc}")
        #print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

    plt.xlabel("Noise level ->")
    plt.ylabel("Absolute Error ->")
    plt.plot(errors, abs_err)
    plt.savefig("file.png")


def evaluate_noise_loss():
    model.eval()

    errors = np.linspace(0,10,200)
    losses = []

    for z in errors:
        pred_times, pred_events = [], []
        gold_times, gold_events = [], []
        loss = []
        for i, batch in enumerate(tqdm(test_loader)):
            batch[0][0] += z
            time_tensor, event_tensor = batch
            time_input, time_target = model.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
            event_input, event_target = model.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

            time_logits, event_logits = model.forward(time_input, event_input)
            loss1 = model.time_criterion(time_logits.view(-1), time_target.view(-1))
            loss.append(loss1.cpu().detach().numpy())
            
        
        #loss = model.Loss(pred_times, gold_times)
        #print(loss[0])
        loss = np.array(loss)
        losses.append(loss.mean())
        print("loss grad := ", loss)
        #acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
        #print(f"epoch {epc}")
        #print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

    plt.xlabel("Noise level ->")
    plt.ylabel("RMTPP Loss ->")
    plt.plot(errors, losses)
    plt.savefig("file_loss.png")


def evaluate_noise_loss_gradient():
    #model.eval()

    errors = np.linspace(0,10,200)
    losses = []

    for z in errors:
        pred_times, pred_events = [], []
        gold_times, gold_events = [], []
        loss = []
        model.train()
        for i, batch in enumerate(tqdm(test_loader)):
            batch[0][0] += z
            time_tensor, event_tensor = batch
            time_input, time_target = model.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
            event_input, event_target = model.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

            time_target.requires_grad = True

            time_logits, event_logits = model.forward(time_input, event_input)
            loss1 = model.time_criterion(time_logits.view(-1), time_target.view(-1))
            loss1.backward()
            #print("loss1 grad wrt time_target :=", time_target.grad)
            #print("grad shape", time_target.grad.shape)
            #quit()

            loss.append(time_target.grad[0].cpu().detach().numpy())


        #loss = model.Loss(pred_times, gold_times)
        #print(loss[0])
        loss = np.array(loss)
        losses.append(loss.mean())
        print("loss gradient := ", loss)
        #acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
        #print(f"epoch {epc}")
        #print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")

    plt.xlabel("Noise level ->")
    plt.ylabel("RMTPP Loss gradient wrt to first target ->")
    plt.plot(errors, losses)
    plt.savefig("file_loss_grad.png")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--model", type=str, default="erpp", help="erpp, rmtpp")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--event_class", type=int, default=7)
    parser.add_argument("--verbose_step", type=int, default=350)
    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    config = parser.parse_args()

    train_set = ATMDataset(config, subset='train')
    test_set = ATMDataset(config, subset='test')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ATMDataset.to_features)

    print("train_set[0] := ", train_set[0])
    #quit()

    weight = np.ones(config.event_class)
    if config.importance_weight:
        weight = train_set.importance_weight()
        print("importance weight: ", weight)
    model = Net(config, lossweight=weight)
    model.set_optimizer(total_step=len(train_loader) * config.epochs, use_bert=True)
    model.cuda()

    for epc in range(config.epochs):
        model.train()
        range_loss1 = range_loss2 = range_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            #print("i := ", i)
            #print("batch[0].shape := ", batch[0].shape)
            #quit()

            l1, l2, l = model.train_batch(batch)
            range_loss1 += l1
            range_loss2 += l2
            range_loss += l

            if (i + 1) % config.verbose_step == 0:
                print("time loss: ", range_loss1 / config.verbose_step)
                print("event loss:", range_loss2 / config.verbose_step)
                print("total loss:", range_loss / config.verbose_step)
                range_loss1 = range_loss2 = range_loss = 0

        evaluate()

    #evaluate_noise()
    evaluate_noise_loss()
    #evaluate_noise_loss_gradient()

import pandas as pd
from models import Transformer
from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from attack_model import FGM,PGD,FreeLB

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from models import Autoformer, Crossformer,DLinear,TiDE,TimesNet,iTransformer,LightTS,Pyraformer,PatchTST,MICN,FiLM,Nonstationary_Transformer,ETSformer
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
upper_limit, lower_limit = 1,0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X,X_mark, dec_inp, batch_y_mark, y, epsilon, alpha, attack_iters, restarts,
               norm= "l_inf", early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None,pred=None):
    
    y=y[:,-pred:,:].to(device)
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            robust=X+delta
            output = model(robust,X_mark ,dec_inp, batch_y_mark)
            
            loss = F.mse_loss(output, y)
            
            loss.backward()
            
            grad = delta.grad.detach()
            d = delta[:, :, :]
            g = grad[:, :, :]
            x = X[:, :, :]
            
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[:, :, :] = d
            delta.grad.zero_()
        all_loss = F.mse_loss(model(X+delta,X_mark,dec_inp, batch_y_mark), y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta
class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'Transformer':Transformer,
            'Informer':Informer,
            'DLinear':DLinear,
            'TimesNet':TimesNet,
            'Crossformer':Crossformer,
            'LightTS':LightTS,
            'iTransformer':iTransformer,
            'Pyraformer':Pyraformer,
            'Autoformer':Autoformer,
            'TiDE':TiDE,
            'MICN':MICN,
            'FiLM':FiLM,
            'Nonstationary_Transformer':Nonstationary_Transformer,
            'ETSformer':ETSformer,
            'PatchTST':PatchTST
        }
        
        if self.args.model=='Informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='Informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers, # self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        else:
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model.to(self.device)
        return model

    def _get_data(self, flag, pre_data=None):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'Sum':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            pre_data=pre_data,
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        # print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag ='train')
        vali_data, vali_loader = self._get_data(flag ='val')
        test_data, test_loader = self._get_data(flag ='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            pgd = PGD(self.model)
            fgm = FGM(self.model)
            K = 3
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                fgm.attack()
                #pgd.backup_grad()
    # 对抗训练
                #for t in range(K):
                 #   pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                 #   if t != K-1:
                 #       model.zero_grad()
                 #   else:
                 #       pgd.restore_grad()

                pred, true = self._process_one_batch(
                      train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss_adv = criterion(pred, true)
                loss_adv.backward()
                fgm.restore()
                #pgd.restore()
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    def train2(self, setting):
        train_data, train_loader = self._get_data(flag ='train')
        vali_data, vali_loader = self._get_data(flag ='val')
        test_data, test_loader = self._get_data(flag ='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            K = 3
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag = 'train')
        fgm = FGM(self.model)
        pgd = PGD(self.model)
        self.model.eval()
        criterion =  self._select_criterion()
        preds = []
        trues = []
        
        #K = 20
        #self.model.eval()
        #delta = torch.zeros_like(x, requires_grad=True)
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x.requires_grad=True
            batch_x_mark.requires_grad=True
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            #loss = criterion(pred, true)
            
            #loss.backward()  # 反向传播得到正确的梯度
            # 进行对抗训练
            
            #pgd.backup_grad()
            #for t in range(K):
             #   pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
             #   if t != K-1:
              #      self.model.zero_grad()
               # else:
               #     pgd.restore_grad()
               # pred, true = self._process_one_batch(
               # test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
           #     loss_adv = criterion(pred, true)
            #    loss_adv.backward()
            #fgm.attack()
           # pred, true = self._process_one_batch(
            #    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred = train_data.inverse_transform(pred)
            true = train_data.inverse_transform(true)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{},rmse:{},mape:{},mspe:{}'.format(mse, mae,rmse,mape,mspe))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        # 增加一个绘图功能

        return
    def test2(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag = 'train')
        fgm = FGM(self.model)
        pgd = PGD(self.model)
        self.model.eval()
        criterion =  self._select_criterion()
        preds = []
        trues = []
        
        #K = 20
        
        #delta = torch.zeros_like(x, requires_grad=True)
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x.requires_grad=True
            batch_x_mark.requires_grad=True
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
            loss = criterion(pred, true)
            
            loss.backward()  # 反向传播得到正确的梯度
            # 进行对抗训练
            
            #pgd.backup_grad()
            #for t in range(K):
             #   pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
             #   if t != K-1:
              #      self.model.zero_grad()
               # else:
               #     pgd.restore_grad()
               # pred, true = self._process_one_batch(
               # test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            
           #     loss_adv = criterion(pred, true)
            #    loss_adv.backward()
            fgm.attack()
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred = train_data.inverse_transform(pred)
            true = train_data.inverse_transform(true)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{},rmse:{},mape:{},mspe:{}'.format(mse, mae,rmse,mape,mspe))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        # 增加一个绘图功能

        return

    def calculate_mse(self, y_true, y_pred):
        # 均方误差
        mse = np.mean(np.abs(y_true - y_pred))
        return mse

    def predict(self, args, setting, Load=True):
        criterion =  self._select_criterion()
        fgm = FGM(self.model)
        history_data = pd.read_csv(args.root_path + args.data_path)[args.target][-args.seq_len:].reset_index(drop=True)
        if args.is_rolling_predict:
            pre_data = pd.read_csv(args.root_path + args.rolling_data_path)
        else:
            pre_data = pd.read_csv(args.root_path + args.data_path)
        pre_data['date'] = pd.to_datetime(pre_data['date'])
        columns = ['forecast' + column for column in pre_data.columns[1:]]
        pre_data.reset_index(inplace=True, drop=True)
        pre_length = args.pred_len
        # 数据都读取进来
        dict_of_lists = {column: [] for column in columns}
        results = []
        for i in range(int(len(pre_data)/pre_length)):
            if i == 0:
                pred_data, pred_loader = self._get_data(flag='pred')
            else:
                pred_data, pred_loader = self._get_data(flag='pred', pre_data=pre_data.iloc[:i*pre_length])

            #print(f'预测第{i + 1} 次')
            if Load:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path+'/'+'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path))

            self.model.eval()

            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = pred_data.inverse_transform(pred)
                if args.features == 'MS' or args.features == 'S':
                    for i in range(args.pred_len):
                        results.append(pred[0][i][0].detach().cpu().numpy())
                else:
                    for j in range(args.enc_in):
                        for i in range(args.pred_len):
                            dict_of_lists[columns[j]].append(pred[0][i][j].detach().cpu().numpy())
                #print(pred)
            if not args.is_rolling_predict:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>不进行滚动预测<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                break

        if not args.is_rolling_predict:
            if args.features == 'MS' or args.features == 'S':
                  df = pd.DataFrame({'forecast{}'.format(args.target): pre_data[args.target]})
                  df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                df.to_csv('Interval-{}'.format(args.data_path), index=False)
        else:
            if args.features == 'MS' or args.features == 'S':
                  df = pd.DataFrame({'date':pre_data['date'], '{}'.format(args.target): pre_data[args.target],
                                     'forecast{}'.format(args.target): pre_data[args.target]})
                  df.to_csv('Interval-{}'.format(args.data_path), index=False)
            else:
                df = pd.DataFrame(dict_of_lists)
                new_df = pd.concat((pre_data,df), axis=1)
                new_df.to_csv('Interval-{}'.format(args.data_path), index=False)
        pre_len = len(dict_of_lists['forecast' + args.target])
        # 绘图
        plt.figure()
        if args.is_rolling_predict:
            if args.features == 'MS' or args.features == 'S':
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), results, label='Predicted Future Values')
            else:
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
                plt.plot(range(len(history_data), len(history_data) + pre_len), dict_of_lists['forecast' + args.target], label='Predicted Future Values')
        else:
            if args.features == 'MS' or args.features == 'S':
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + len(results)), results,
                         label='Predicted Future Values')
            else:
                plt.plot(range(len(history_data)), history_data,
                         label='Past Actual Values')
                plt.plot(range(len(history_data), len(history_data) + len(dict_of_lists['forecast' + args.target])),
                         dict_of_lists['forecast' + args.target], label='Predicted Future Values')
        # 添加图例
        plt.legend()
        plt.style.use('ggplot')
        # 添加标题和轴标签
        plt.title('Past vs Predicted Future Values')
        plt.xlabel('Time Point')
        plt.ylabel('Value')
        # 在特定索引位置画一条直线
        plt.axvline(x=len(history_data), color='blue', linestyle='--', linewidth=2)
        # 显示图表
        plt.savefig('forcast.png')
        plt.show()
        return
        
    def _process_one_batch_vali_pgd(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        #std = torch.tensor((1.0)).view(1, 1).to(self.device)
        epsilon = (8 / 255.)
        alpha = (2 / 255.)


        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        pgd_delta = attack_pgd(self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y, epsilon, alpha, 3, 1,pred=self.args.pred_len)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)
        
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y =dataset_object.inverse_transform(batch_y)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        

        return outputs, batch_y
    def _process_one_batch_vali_pgd2(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        #std = torch.tensor((1.0)).view(1, 1).to(self.device)
        epsilon = (8 / 255.)
        alpha = (2 / 255.)


        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        pgd_delta = attack_pgd(self.model, batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y, epsilon, alpha,1, 1,pred=self.args.pred_len)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x+pgd_delta, batch_x_mark, dec_inp, batch_y_mark)
        
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y =dataset_object.inverse_transform(batch_y)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        

        return outputs, batch_y
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        

        return outputs, batch_y

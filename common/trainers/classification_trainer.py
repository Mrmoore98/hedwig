import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.trainers.trainer import Trainer
from models.oh_cnn_HAN.check_text import match_str
from models.oh_cnn_HAN.loss import Loss
from models.oh_cnn_HAN.label_smooth import LabelSmoothing

# #decoder
# from BCPGDS_decoder.Read_IMDB import Load_Data
# from BCPGDS_decoder.Config_for_decoder import decoder_setting
# from BCPGDS_decoder.Update_decoder import updatePhi_Pi
# from BCPGDS_decoder.Config import Setting, SuperParams, Params, Data

class ClassificationTrainer(Trainer):

    def __init__(self, model, embedding, train_loader, trainer_config, config_main, train_evaluator, test_evaluator, dev_evaluator):
        super().__init__(model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)
        self.config = trainer_config
        self.config_main = config_main
        self.early_stop = False
        self.best_dev_f1 = 0
        self.iterations = 0
        self.iters_not_improved = 0

        
        self.start = None
        self.model_time = 0
        self.learning_time = 0
        self.dev_time = 0
        self.model_process_time = 0
        self.initialize_time = 0
        self.initial_data_time = 0
        self.batch_time = 0

        self.log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f}'.split(','))
        self.dev_log_template = ' '.join(
            '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.4f},{:>8.4f},{:8.4f},{:12.4f},{:12.4f},{:>8.4f}'.split(','))
        
        self.train_result = []
        self.dev_result = [] #[fig,fig,list]

        self.label_smoothing = LabelSmoothing(config_main).cuda()
        self.Loss = Loss(config_main).cuda()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.model_outfile, self.train_loader.dataset.NAME, '%s.pt' % timestamp)

        self.Params=None 
        self.Data=None 
        self.SuperParams=None
        self.epsit =None

    def train_epoch(self, epoch):
        itmp = time.time() 
        self.train_loader.init_epoch()
        self.initial_data_time = time.time()-itmp
        tmp_batch = time.time()

        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            
            
            self.batch_time += time.time() - tmp_batch 
            initialize_time_tmp = time.time()
            
            self.iterations += 1
            self.model.train()
            if self.config['optimizer_warper']:
                self.optimizer.optimizer.zero_grad() #for warper
            else:
                self.optimizer.zero_grad() #origin
            
            # if batch.text.shape[2]*batch.text.shape[1]> 100:
            #     import pdb; pdb.set_trace()
            #     match_data = match_str(batch.text,batch.dataset.fields['text'].vocab.itos)
            self.initialize_time += time.time()-initialize_time_tmp
            model_process_time_tmp = time.time()    
            
            # try:
            if hasattr(self.model, 'tar') and self.model.tar:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    scores, rnn_outs = self.model(batch.text)
                else:
                    scores, rnn_outs = self.model(batch.text[0], lengths=batch.text[1])
            else:
                if 'ignore_lengths' in self.config and self.config['ignore_lengths']:
                    if self.config_main.vae_struct:
                        scores, feature_map = self.model(batch.text[0])
                    else:
                        scores = self.model(batch.text)
                else:
                    scores = self.model(batch.text[0], lengths=batch.text[1])
            if self.config_main.vae_struct and self.iterations % 50 == 1:
                [self.Params.D1_k1, self.Params.Pi_left, self.Params.Pi_right] = updatePhi_Pi(epoch, batch.text[1], self.Params, self.Data, self.SuperParams, batch_idx, self.Setting, feature_map[:,:,:,:self.config_main.word_num_hidden], 
                            feature_map[:,:,:,self.config_main.word_num_hidden:], self.epsit)


            self.model_process_time += time.time()-model_process_time_tmp
            learning_tmp = time.time()
            # except RuntimeError:
            #     import pdb; pdb.set_trace()

            if 'is_multilabel' in self.config and self.config['is_multilabel']:
                predictions = F.sigmoid(scores).round().long()
                for tensor1, tensor2 in zip(predictions, batch.label):
                    if np.array_equal(tensor1, tensor2):
                        n_correct += 1
                loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
            else:
                    
                if self.config['loss'] is not None:
                    if self.config_main.vae_struct:
                        if len(scores.shape)<2:
                            scores = scores.unsqueeze(0)
                        predictions  = torch.argmax(scores, dim=-1).long().cpu().numpy()
                        ground_truth = torch.argmax(label, dim=-1).cpu().numpy()
                        n_correct   += np.sum(predictions == ground_truth)
                        loss, n_correct = Loss(self.config['loss'])(scores, batch.label.data, feature_map, batch.text)
                    else:
                        loss, n_correct = Loss(self.config['loss'])(scores, batch.label.data)

                elif self.config['Binary']:
                    predictions  = F.sigmoid(scores).round().long().cpu().numpy()
                    ground_truth = torch.argmax(batch.label.data, dim=1).cpu().numpy()
                    n_correct   += np.sum(predictions == ground_truth)

                    loss = F.binary_cross_entropy_with_logits(scores, torch.argmax(batch.label.data, dim=1).type(torch.cuda.FloatTensor))
                else:
                
                    if len(scores.shape)<2:
                        scores = scores.unsqueeze(0)
                    predictions  = torch.argmax(scores, dim=-1).long().cpu().numpy()
                    ground_truth = torch.argmax(batch.label.data, dim=-1).cpu().numpy()
                    n_correct   += np.sum(predictions == ground_truth)

                    if self.config_main.label_smoothing:
                        loss = self.label_smoothing(scores,torch.argmax(batch.label.data, dim=1))
                    else:
                        loss = F.cross_entropy(scores, torch.argmax(batch.label.data, dim=1))
                

            if hasattr(self.model, 'tar') and self.model.tar:
                loss = loss + self.model.tar * (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()
            if hasattr(self.model, 'ar') and self.model.ar:
                loss = loss + self.model.ar * (rnn_outs[:]).pow(2).mean()

            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            loss.backward()
            self.optimizer.step()
            try:
                self.config_main.schedular.step(epoch + batch_idx/len(self.train_loader))
            except:
                pass

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                self.model.update_ema()

            if self.iterations % 300 == 1:
                niter = epoch * len(self.train_loader) + batch_idx
                print(self.log_template.format(time.time() - self.start, epoch, self.iterations, 1 + batch_idx,
                                               len(self.train_loader), 100.0 * (1 + batch_idx) / len(self.train_loader),
                                               loss.item(), train_acc))
                self.train_result.append((train_acc,loss.item()))
            # if self.iterations % (300*3) == 1:
            #     dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, mse = self.dev_evaluator.get_scores()[0]
            #     print(self.dev_log_template.format(time.time() - self.start, 1, self.iterations, 1, 1,
            #                                                 dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, mse))

                                
            self.learning_time += time.time()-learning_tmp
            tmp_batch = time.time()

    def train(self, epochs):
        self.start = time.time()
        header = '  Time Epoch Iteration Progress    (%Epoch)   Loss     Accuracy'
        dev_header = '  Time Epoch Iteration Progress     Dev/Acc. Dev/Pr.  Dev/Recall   Dev/F1       Dev/Loss'
        os.makedirs(self.model_outfile, exist_ok=True)
        os.makedirs(os.path.join(self.model_outfile, self.train_loader.dataset.NAME), exist_ok=True)
        if self.config_main.vae_struct:
            #decoder
            from BCPGDS_decoder.Read_IMDB import Load_Data
            from BCPGDS_decoder.Config_for_decoder import decoder_setting
            from BCPGDS_decoder.Update_decoder import updatePhi_Pi
            from BCPGDS_decoder.Config import Setting, SuperParams, Params, Data
            [self.Data, self.Setting, self.Params, self.SuperParams, self.epsit] = decoder_setting(self.config_main.decoder_dataset, self.config_main)
        for epoch in range(1, epochs + 1):
            print('\n' + header)
            self.train_epoch(epoch)
            print("Performing evluation!")
            # Evaluate performance on validation set
            dev_time_tmp = time.time()
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, mse = self.dev_evaluator.get_scores()[0]
            self.dev_time += time.time() - dev_time_tmp
            # Print validation results
            print('\n' + dev_header)
            print(self.dev_log_template.format(time.time() - self.start, epoch, self.iterations, epoch, epochs,
                                               dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, mse))
            print("tot:{:.0f}, batch{:.0f}, model:{:.0f}, learning:{:.0f}, dev:{:.0f}, initial:{:.0f}, initial_data{:.0f}".format(time.time() - self.start, self.batch_time, self.model_process_time, self.learning_time, self.dev_time, self.initialize_time, self.initial_data_time))
            
            self.dev_result.append([dev_acc, dev_loss, time.time() - self.start])
            
            # Update validation results
            if dev_f1 > self.best_dev_f1:
                self.iters_not_improved = 0
                self.best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)
            else:
                self.iters_not_improved += 1
                if self.iters_not_improved >= self.patience:
                    self.early_stop = True
                    print("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                    break

        return self.dev_result, self.train_result

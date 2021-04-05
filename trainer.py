import os
import math
from decimal import Decimal

import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.testscale = args.testscale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        if self.model.training:
            self.writer = SummaryWriter(os.path.join(ckp.dir, 'log'))
        self.count = 0


    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)

        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        h_times_w_offset_coord = h_offset_coord * w_offset_coord
        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        h_times_w_offset_coord = h_times_w_offset_coord.contiguous().view(1, -1,1)
        pos_mat2 = pos_mat ** 2

        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)
        pos_mat = torch.cat([pos_mat,h_times_w_offset_coord, pos_mat2], 2)
        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            N,C,H,W = lr.size()
            _,_,outH,outW = hr.size()
            scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.scale[idx_scale])  ###  get the position matrix, mask

            if self.args.n_GPUs>1 and not self.args.cpu:
                scale_coord_map = torch.cat([scale_coord_map]*self.args.n_GPUs,0)
            else:
                scale_coord_map = scale_coord_map.to(device)
            
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale, scale_coord_map)
            re_sr = torch.masked_select(sr,mask.to(device))
            re_sr = re_sr.contiguous().view(N,C,outH,outW)
            loss = self.loss(re_sr, hr)
            self.writer.add_scalar('loss', loss, self.count)
            self.count += 1
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        if self.args.n_GPUs == 1:
            target = self.model
        else:
            target = self.model   #.module

        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir,'model', 'model_{}.pt'.format(epoch))
        )
        ## save models
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.ckp.dir, 'optimizer.pt')
        )

    def test(self):  
        epoch = self.scheduler.last_epoch + 1
        self.scheduler.last_epoch = epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.testscale)))
        self.model.eval()
        timer_test = utility.timer()
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        ssims = []
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.testscale):
                eval_acc = 0
                eval_acc_ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    N,C,H,W = lr.size()
                    scale = self.args.testscale[idx_scale]
                    outH,outW = int(H*scale),int(W*scale)
                    #### When the GPU is enough to hold the whole test image, test it directly ###
                    scale_coord_map, mask = self.input_matrix_wpn(H,W,self.args.testscale[idx_scale])
                    scale_coord_map = scale_coord_map.to(device)
                    output = self.model(lr, idx_scale, scale_coord_map)
                    output = torch.masked_select(output,mask.to(device))
                    output = output.contiguous().view(N,C,outH,outW)
                    ### When It's difficult to test a whole image because of heavy memory consumption, we can split it to test. ####
                    # timer_test.tic()
                    # output = lr.new(N, C, outH, outW)
                    # for i in range(0, H, 128):
                    #     for j in range(0, W, 128):
                    #         lr_patch = lr[:,:,i:i+128, j:j+128]
                    #         scale_coord_map, mask = self.input_matrix_wpn(lr_patch.shape[2],lr_patch.shape[3],self.args.testscale[idx_scale])
                    #         scale_coord_map = scale_coord_map.to(device)
                    #         sr = self.model(lr_patch, idx_scale, scale_coord_map)
                    #         sr = torch.masked_select(sr,mask.to(device))
                    #         re_sr = sr.contiguous().view(N,C,int(lr_patch.shape[2]*scale),int(lr_patch.shape[3]*scale))
                    #         output[:,:,int(i*scale):int(i*scale)+int(lr_patch.shape[2]*scale), int(j*scale):int(j*scale)+int(lr_patch.shape[3]*scale)] = re_sr

                    # timer_test.hold()
                    ##################################################
                    sr = utility.quantize(output, self.args.rgb_range)
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_acc_ssim += utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                ssims.append(eval_acc_ssim/len(self.loader_test))
            ##################### record results #################        
            best = self.ckp.log.max(0)
            for idx_scale in range(len(self.testscale)):
                self.ckp.write_log(
                    '[{} x{} ]\tPSNR: {:.3f} SSIM: {:.4f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        self.testscale[idx_scale],
                        self.ckp.log[-1, idx_scale],
                        ssims[idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
                
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.save(self.ckp.log, os.path.join(self.ckp.dir,'psnr.pt'))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


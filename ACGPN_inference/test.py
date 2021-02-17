import os
import time

import cv2
import numpy as np
import torch
import util.util as util
from models.models import create_model
from options.train_options import TrainOptions
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from data.data_loader import CreateDataLoader
from models.pix2pixHD_model import generate_discrete_label

writer = SummaryWriter("runs/G1G2")
SIZE = 320
NC = 14


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img * (1 - mask)
    M_c = (1 - mask.cuda()) * M_f
    M_c = M_c + torch.zeros(img.shape).cuda()  ##broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    # check=check>0
    # print(check)
    masked_label = label * (1 - mask)
    masked_edge = mask * edge
    masked_color_strokes = mask * (1 - color_mask) * color
    masked_noise = mask * noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((data["label"].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data["label"].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data["label"].cpu().numpy() == 7).astype(np.int))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label


opt = TrainOptions().parse()
os.makedirs(opt.results_dir, exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, "iter.txt")
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=",", dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print("Resuming from epoch %d at iteration %d" % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print("# Inference images = %d" % dataset_size)

model = create_model(opt)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

step = 0

ce_loss_values = {}

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        # save_fake = total_steps % opt.display_freq == display_delta
        save_fake = True

        ##add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((data["label"].cpu().numpy() == 7).astype(np.float))
        #
        # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor((data["label"].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((data["label"].cpu().numpy() > 0).astype(np.int))
        img_fore = data["image"] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data["label"])

        ############## Forward Pass ######################
        (
            losses,
            # fake_image,
            real_image,
            G1_in,
            input_label,
            input_label_arm_refined,
            L1_loss,
            # style_loss,
            # clothes_mask,
            CE_loss,
            rgb,
            # alpha,
        ) = model(
            Variable(data["label"].cuda()),  # label
            Variable(data["edge"].cuda()),  # pre_clothes_mask
            Variable(img_fore.cuda()),  # img_fore
            Variable(mask_clothes.cuda()),  # clothes_mask
            Variable(data["color"].cuda()),  # clothes
            Variable(all_clothes_label.cuda()),  # all_clothes_label
            Variable(data["image"].cuda()),  # real_image
            Variable(data["pose"].cuda()),  # pose
            Variable(data["image"].cuda()),  # grid
            Variable(mask_fore.cuda()),  # mask_fore
        )

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict["D_fake"] + loss_dict["D_real"]) * 0.5
        loss_G = loss_dict["G_GAN"] + torch.mean(
            CE_loss
        )  # loss_dict.get('G_GAN_Feat',0)+torch.mean(L1_loss)+loss_dict.get('G_VGG',0)

        writer.add_scalar("loss_d", loss_D, step)
        writer.add_scalar("loss_g", loss_G, step)
        # writer.add_scalar('loss_L1', torch.mean(L1_loss), step)

        writer.add_scalar("loss_CE", torch.mean(CE_loss), step)
        # writer.add_scalar('acc', torch.mean(acc)*100, step)
        # writer.add_scalar('loss_face', torch.mean(face_loss), step)
        # writer.add_scalar('loss_fore', torch.mean(fore_loss), step)
        # writer.add_scalar('loss_tv', torch.mean(tv_loss), step)
        # writer.add_scalar('loss_mask', torch.mean(mask_loss), step)
        # writer.add_scalar('loss_style', torch.mean(style_loss), step)

        writer.add_scalar("loss_g_gan", loss_dict["G_GAN"], step)
        # writer.add_scalar('loss_g_gan_feat', loss_dict['G_GAN_Feat'], step)
        # writer.add_scalar('loss_g_vgg', loss_dict['G_VGG'], step)

        ############### Backward Pass ####################
        # update generator weights
        # model.module.optimizer_G.zero_grad()
        # loss_G.backward()
        # model.module.optimizer_G.step()
        #
        # # update discriminator weights
        # model.module.optimizer_D.zero_grad()
        # loss_D.backward()
        # model.module.optimizer_D.step()

        # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ############## Display results and errors ##########

        ### display output images

        ce_loss_values[data["name"][0]] = float(CE_loss.detach().cpu())
        for name, tensor, put_palette in (
            ("source_image", data["image"], False),
            ("source_label", generate_label_color(data["label"]), False),

            ("G1_in_1", G1_in[:, 0].detach().cpu(), False),
            ("G1_in_1", G1_in[:, 0].detach().cpu(), False),
            ("G1_in_2", G1_in[:, 1:4].detach().cpu(), False),
            ("G1_in_3", generate_label_color(generate_label_plain(G1_in[:, 4:4 + 14])).detach().cpu(), False),
            ("G1_in_4", (generate_discrete_label(G1_in[:, 4 + 14:-1].detach(), 18, False) > 0).cpu(), False),
            ("G1_in_5", G1_in[:, -1].squeeze().detach().cpu(), False),

            ("G1_out", generate_label_color(generate_label_plain(input_label)).detach().cpu(), False),
            ("G1_gt", generate_label_color(
                (data["label"] * (1 - mask_clothes))
            ), False)
        ):
            d_image = tensor.squeeze().numpy().copy()
            if len(d_image.shape) == 3:
                d_image = np.moveaxis(d_image, 0, -1)
            if not put_palette:
                d_image = (d_image + 1.) / 2. * 255
            d_image = d_image.astype(np.uint8)
            # print(image.shape, image.dtype)

            base_filename = os.path.splitext(data["name"][0])
            output_name = f"{base_filename}_{name}.jpg"

            imageio.imwrite(opt.results_dir + "/" + output_name, d_image, quality=97)

        a = generate_label_color(generate_label_plain(input_label)).float().cuda()
        b = real_image.float().cuda()
        c = fake_image.float().cuda()
        d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
        combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze()
        # combine=c[0].squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        if step % 1 == 0:
            writer.add_image("combine", (combine.data + 1) / 2.0, step)
            rgb = (cv_img * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            n = str(step) + ".jpg"
            cv2.imwrite(opt.results_dir + "/" + data["name"][0], bgr)
        step += 1
        print(step, time.time() - epoch_start_time, (time.time() - epoch_start_time) / step)
        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            # np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            pass
        if epoch_iter >= dataset_size:
            break

    with open(os.path.join(opt.results_dir, "ce_losses.json"), "w") as f:
        json.dump(ce_loss_values, f)

    # end of epoch
    iter_end_time = time.time()
    print(
        "End of epoch %d / %d \t Time Taken: %d sec"
        % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
    )
    break

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print("saving the model at the end of epoch %d, iters %d" % (epoch, total_steps))
        model.module.save("latest")
        model.module.save(epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

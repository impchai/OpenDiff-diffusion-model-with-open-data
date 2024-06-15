import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance
import math

def print_cuda_memory_usage():
    # 当前GPU的索引，默认为0
    device = torch.cuda.current_device()
    # 将bytes转换为MB
    bytes_to_mb = lambda x: x / 1024 / 1024
    # 已分配的显存
    allocated = bytes_to_mb(torch.cuda.memory_allocated(device))
    # 显存的总量
    max_allocated = bytes_to_mb(torch.cuda.max_memory_allocated(device)) # 峰值
    # 当前保留的显存总量（包括空闲的）
    reserved = bytes_to_mb(torch.cuda.memory_reserved(device))
    # 最大保留的显存总量（包括空闲的，峰值）
    max_reserved = bytes_to_mb(torch.cuda.max_memory_reserved(device))

    return max_allocated
    # print(f"Allocated memory: {allocated:.2f} MB")
    # print(f"Max allocated memory: {max_allocated:.2f} MB")
    # print(f"Reserved memory: {reserved:.2f} MB")
    # print(f"Max reserved memory: {max_reserved:.2f} MB")



def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=50,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[p1, p2],
                                                        gamma=0.1)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no +
                                         1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0,
                          maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss":
                                avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model,
             test_loader,
             nsample=100,
             scaler=1,
             mean_scaler=0,
             foldername=""):
    print('testnsample=', nsample)
    with torch.no_grad():  # 临时禁用梯度计算
        model.eval()
        evalpoints_total = 0
        evalpoints_ssim = 0
        js_total = 0
        js_one_total = 0
        eval_js_total = 0
        evalpoints_one_total = 0
        ssim_value = 0
        tv_distance_total = 0
        all_target = []
        all_observed_time = []
        all_generated_samples = []
        cut = 5
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                sample, c_targets, observed_time = output


                samples = sample[:, :, cut:-cut]
                c_target = c_targets[:, cut:-cut]
                B, L= c_target.shape

                samples_median = samples.median(dim=1)[0]

                samples_mean = samples_median.mean(dim=0)
                c_target_mean = c_target.mean(dim=0)

                all_target.append(c_target)

                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)


                #----------------------------generated Metric------------------------------------
                # #：Metrix 1----JS Divers


                flatten1_samp = samples_mean.cpu().numpy().flatten()
                flatten1_targ = c_target_mean.cpu().numpy().flatten()

                aaa1 = (flatten1_samp - flatten1_samp.min()) / (
                    flatten1_samp.max() - flatten1_samp.min())
                norm1_samp = aaa1 / aaa1.sum()
                bbb1 = (flatten1_targ - flatten1_targ.min()) / (
                    flatten1_targ.max() - flatten1_targ.min())
                norm1_targ = bbb1 / bbb1.sum()
                js_distance = distance.jensenshannon(norm1_samp, norm1_targ,
                                                     2.0)

                js_total += js_distance.item()
                #
                # #：Metrix 2----1-阶 JS Divers
                gen_dif_one = samples_mean[1:] - samples_mean[:-1]
                tar_dif_one = c_target_mean[1:] - c_target_mean[:-1]

                flatten2_samp = gen_dif_one.cpu().numpy().flatten()
                flatten2_targ = tar_dif_one.cpu().numpy().flatten()

                aaa2 = (flatten2_samp - flatten2_samp.min()) / (
                    flatten2_samp.max() - flatten2_samp.min())
                norm2_samp = aaa2 / aaa2.sum()
                bbb2 = (flatten2_targ - flatten2_targ.min()) / (
                    flatten2_targ.max() - flatten2_targ.min())
                norm2_targ = bbb2 / bbb2.sum()

                js_distance_dif_one = distance.jensenshannon(
                    norm2_samp, norm2_targ, 2.0)

                js_one_total += js_distance_dif_one.item()
                eval_js_total += 1
                evalpoints_one_total += B

                # #：Metrix 3----TV-Distance
                tv_distance = 0

                for i in range(len(c_target)):
                    tv_distance += 0.5 * abs(samples_median[i] - c_target[i])
                
                tv_distance_res = tv_distance / samples_median.shape[1]
                tv_distance_total += tv_distance_res.sum().item()

                ssim_value = 1
                evalpoints_ssim = 1
                # ----------------------------generated Metric------------------------------------

                it.set_postfix(
                    ordered_dict={
                        "js_total": js_total / eval_js_total,
                        "js_one_total": js_one_total / eval_js_total,
                        "tv_distance":
                        tv_distance_total / evalpoints_one_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) +
                    ".pk", "wb") as f:
                all_target = torch.cat(all_target, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(all_target, all_generated_samples,
                                      mean_scaler, scaler)

            with open(foldername + "/result_nsample" + str(nsample) + ".pk",
                      "wb") as f:
                pickle.dump(
                    [
                        js_total / eval_js_total,
                        js_one_total / eval_js_total,
                        tv_distance_total / evalpoints_one_total,
                        CRPS,
                        ssim_value / evalpoints_ssim,
                    ],
                    f,
                )
                print("JS_div:", js_total / eval_js_total)
                print("JS_one_div:", js_one_total / eval_js_total)
                print("tv-distance:", tv_distance_total / evalpoints_one_total)
                print("CRPS:", CRPS)

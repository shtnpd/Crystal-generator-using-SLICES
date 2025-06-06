import sys
import torch
import torch.multiprocessing as mp
import os
from slices.core import SLICES
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from multiprocessing import Process, Queue
from tqdm import tqdm
import traceback
import logging

logging.getLogger().setLevel(logging.CRITICAL + 1) 

def slices_subjob(jobs_queue, return_queue):
    backend = SLICES(relax_model="nope", requires_min=False)

    while True:
        new_job = jobs_queue.get()
        try:
            backend.SLICES2structure(new_job)
            return_queue.put((new_job, True))
        except Exception as e:
            # traceback.print_exc()
            return_queue.put((new_job, False))

def inference_subjob(bg_range, fe_range, gpu, checkpoint_path, tokens_queue, num_samples):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    CHECKPOINT_DIR = f"checkpoints/{checkpoint_path}"

    tokenizer = GPT2TokenizerFast.from_pretrained(CHECKPOINT_DIR)
    # model     = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
    model = torch.compile(
        GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16)
              .to("cuda"),
        mode="reduce-overhead"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = torch.device("cuda")
    model.to(device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        for bg_v in bg_range:
            bg = bg_v / 10
            for fe_v in fe_range:
                fe = fe_v / 10
                prompt_bg = f"<BG_{bg}>"
                prompt_fe = f"<FE_{fe}>"

                prompt = "<BOS> " + prompt_bg + " " + prompt_fe + " |"

                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=False
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                max_length = 300
                num_samples = num_samples
                do_sample = True
                top_p = 0.9
                top_k = 50
                temperature = 0.7

                eos_id = tokenizer.eos_token_id

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,

                    max_length=max_length,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                )

                for i, out_ids in enumerate(outputs):
                    generated = tokenizer.decode(out_ids, skip_special_tokens=False)
                    if "<EOS>" in generated:
                        generated = generated.split("<EOS>")[0] + "<EOS>"
                    if "|" in generated:
                        parts = generated.split("|", 1)
                        slices_str = parts[1]
                        slices_str = slices_str.replace("<EOS>", "").strip()
                    tokens_queue.put(slices_str)

        

def job(bg_range, fe_range, checkpoint_path, gpu, queue, worker_id, jobs, num_samples, save_folder):
    tokens_queue = Queue()
    return_queue = Queue()

    processes = []
    for i in range(jobs):
        p = Process(target=slices_subjob, args=(tokens_queue, return_queue))
        processes.append(p)
        p.start()

    gpu_job = Process(target=inference_subjob, args=(bg_range, fe_range, gpu, checkpoint_path, tokens_queue, num_samples))
    gpu_job.start()

    for bg_v in bg_range:
        bg = bg_v / 10
        for fe_v in fe_range:
            fe = fe_v / 10
            if os.path.exists(save_folder + "/" + str(bg) + "_" + str(fe) + ".csv"):
                queue.put((None, ))
                continue
            with open(save_folder + "/" + str(bg) + "_" + str(fe) + ".csv", "w", encoding="utf-8") as fout:
                for out in range(num_samples):
                    res = return_queue.get()
                    if res[1]:
                        # backend.SLICES2structure(slices_str)
                        # print("SUCCESS!")
                        fout.write(f"1,{res[0]}\n")
                        # full_results.write(f"1,{bg},{fe},{slices_str}\n")
                        # correct += 1
                        queue.put((1, f"1,{bg},{fe},{res[0]}\n"))
                        # fout.write("SUCCESS!\n")
                    else:
                        # print("FAIL!", e)
                        fout.write(f"0,{res[0]}\n")
                        # full_results.write(f"0,{bg},{fe},{slices_str}\n")
                        queue.put((0, f"0,{bg},{fe},{res[0]}\n"))
                        # fout.write("FAIL! " + str(e) + "\n")
                    # finally:
                        # total += 1
                        # pbar.update(1)
                        # pbar.set_postfix({"correct": correct, "total": total, "bg": f"{bg:.2f}", "fe": f"{fe:.2f}", "accuracy": f"{correct / total * 100:.2f}%"})

    gpu_job.terminate()
    gpu_job.join()

    for p in processes:
        p.terminate()
        p.join()
    


def infernece_exp(bg_range, fe_range, checkpoint_path="finetune_gpt2_slices/checkpoint-4800", save_folder="inference-results", num_samples=100, gpus=4, jobs=4):
    correct = 0
    total = 0

    queue = Queue()
    processes = []
    for i in range(gpus):
        num_gpus = gpus
        chunks = [bg_range[i::num_gpus] for i in range(num_gpus)]
        p = Process(target=job, args=(chunks[i], fe_range, checkpoint_path, i, queue, i, jobs, num_samples, save_folder))
        processes.append(p)
        p.start()

    with open(save_folder + "/results.csv", "a", encoding="utf-8", buffering=1<<10) as full_results:
        with tqdm(total=len(bg_range) * len(fe_range) * num_samples, smoothing=0.05) as pbar:
            res = queue.get()
            pbar.reset()
            if res[0] is None:
                pbar.update(num_samples)
            else:
                correct += res[0]
                total += 1
                full_results.write(res[1])
                pbar.update(1)
                pbar.set_postfix({"correct": correct, "total": total, "accuracy": f"{correct / total * 100:.2f}%"})
            for i in range(len(bg_range) * len(fe_range) * num_samples - 1):
                res = queue.get()
                if res[0] is None:
                    pbar.update(num_samples)
                    continue
                correct += res[0]
                total += 1
                full_results.write(res[1])
                pbar.update(1)
                pbar.set_postfix({"correct": correct, "total": total, "accuracy": f"{correct / total * 100:.2f}%"})

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # infernece_exp(range(51), range(-50, 51), checkpoint_path="finetune_gpt2_slices_5/checkpoint-1800", num_samples=100, gpus=2, jobs=16)
    infernece_exp(range(51), range(-50, 0), checkpoint_path="finetune_gpt2_slices_5/checkpoint-1800", num_samples=100, gpus=2, jobs=14)
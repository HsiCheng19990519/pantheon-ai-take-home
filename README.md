<div align="center">

# Pantheon Lab Programming Assignment

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Interview Questions

### 1) What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.
The discriminator \(D\) provides the **training signal** that drives the generator \(G\) toward the data distribution. Concretely, \(D\) maps an input image (and an optional label for conditional GANs) to a scalar **real/fake** score. During training, \(D\) is optimized to increase this score for real data and decrease it for generated data; \(G\) is optimized to produce samples that **maximize** the discriminator’s score (i.e., “look real” to \(D\)). In short, \(D\) acts as a learned, adaptive **critic** whose gradients push \(G\) to reduce the discrepancy between \(p_\text{data}\) and \(p_G\).

---

### 2) The generator network in this code base takes two arguments: `noise` and `labels`. What are these inputs and how could they be used at inference time to generate an image of the number 5?
- **Noise** \(\mathbf{z}\): a latent vector sampled from a simple prior (e.g., \(\mathcal{N}(0,I)\)). It injects stochasticity so the generator can produce diverse outputs.
- **Labels** \(y\): class-conditioning information (e.g., digits \(0\!-\!9\)). In a conditional GAN, \(G\) takes \((\mathbf{z}, y)\) and learns class-specific manifolds.

**To generate a “5”**: set \(y=5\) and sample multiple \(\mathbf{z}\) vectors; feed \((\mathbf{z}, y{=}5)\) into \(G\). We will get diverse images that all depict the digit ‘5’.

---

### 3) What steps are needed to deploy a model into production?
1. **Define the contract**: inputs/outputs, quality & latency SLOs, safety/abuse constraints.
2. **Package the model**: freeze/export or containerize a Python service with all preprocessing/postprocessing.
3. **Serve**: expose via REST/gRPC; add batching/queuing; choose accelerators; scale horizontally (autoscaling) and vertically (FP16/INT8, TensorRT/ONNX Runtime).
4. **Observability**: structured logs, metrics (latency, throughput, error rate), health checks, tracing, drift detection.
5. **Release strategy**: shadow traffic, canary, rollback plans; version datasets and artifacts.
6. **Security & compliance**: authn/authz, rate limits, PII handling, audit logging.
7. **Lifecycle**: continuous evaluation, data collection & re-training, automated tests (unit/integration/regression).

---

### 4) If you wanted to train with multiple GPUs, what can you do in pytorch lightning to make sure data is allocated to the correct GPU?
- Use **DDP** (or a DDP variant) so Lightning launches one process per GPU and synchronizes gradients correctly.
- Rely on **DistributedSampler** (Lightning can set it automatically) to ensure each rank sees a **disjoint shard** each epoch; call `set_epoch(epoch)` so shuffling differs across epochs.
- Avoid rank-dependent side effects in `training_step`; move tensors to the current device; keep RNG seeding deterministic when needed.
- **Log/Checkpoint only on rank 0** to prevent duplication/corruption.
- Validate device placement and batch shapes per rank; don’t perform CPU/GPU copies inside tight loops.

---

## W&B Evidence (graphs & images)

**Project:** `https://wandb.ai/hsicheng/Tests/workspace?nw=nwuserhsicheng19990519`
My results in different stages are from the codes in following branches, respectively.

### `baseline/v1`
- Results for basic assignments without open-ended tasks.
- https://api.wandb.ai/links/hsicheng/idvbcnd6

### `cnn-update/v1`
- Results for MNIST datasets and G/D with CNN.
- https://api.wandb.ai/links/hsicheng/xe6d89on

### `color-update/v1`
- Results for CIFAR-10 datasets (64*64 resolution and RGB colors) and G/D with CNN.
- https://wandb.ai/hsicheng/Tests/reports/color-update-v1--VmlldzoxNDAyNzEyNg?accessToken=xtyc7f37f1ldkxm7c13xm6ium5pbu05oixqykqql43q7rjma5h77vgbutjysf2fb

### `loss-update/v1`
- Results for CIFAR-10 DATASETS and G/D with CNN, with hinge loss + TTUR + EMA + SN (failed due to I do not have time to tune the parameters before the deadline).
- https://wandb.ai/hsicheng/Tests/reports/loss-update-v1--VmlldzoxNDAyNzE1MA?accessToken=emwcjoh23kyxe8hmvy6l12idcpqkw0hqnw2012zqde00lzgtossixai0iiwmcc9w

## Difficulties I encountered and how I overcame them

### 1) Moving from academic scripts to project-level hygiene
- **Symptom:** Coming from physics and a 2-month internship, I wasn’t fully fluent with multi-file projects, Hydra configs, Lightning lifecycles, and reproducible runs.
- **Action** I read the repo’s structure carefully, set up small, verifiable increments on separate branches, and used Hydra overrides for all experiments.
- **Result:** A baseline that trains deterministically and a clean path to compare/rollback between variants.

### 2) Hydra/OmegaConf interpolation errors (`${n_classes}` not found)
- **Symptom:** Runs crashed during config printing (`InterpolationKeyError`).
- **Diagnosis:** Generator/Discriminator blocks referenced parent keys with `${n_classes}` instead of `${..n_classes}` (relative interpolation).
- **Action:** Fixed all nested references to `${..n_classes}`, `${..latent_dim}`, `${..channels}`, `${..img_size}`.
- **Result:** Configs render correctly; experiments start reliably.

### 3) W&B run marked as “failed” after successful training
- **Symptom:** Training completed and images were logged, but the run showed **failed**.
- **Diagnosis:** `trainer.test()` was invoked without explicitly passing `model/datamodule` on this Lightning version; test loop validation failed.
- **Action:** Called `trainer.test(model=model, datamodule=datamodule, ckpt_path=None)` and added a guard to skip testing if a valid test loader is unavailable.
- **Result:** Clean **finished** runs; artifacts and logs preserved.

### 4) CNN made training much slower
- **Symptom:** After switching to CNN, epochs were significantly slower on CPU.
- **Action:** Moved training to the my personal GPU.
- **Result:** Throughput acceptable; training remains stable for both MNIST and CIFAR-10.

### 5) Reproducibility & rollback
- **Symptom:** It’s easy to “improve” something and accidentally break earlier working pieces.
- **Action:** Kept each milestone on its own branch (`baseline/v1`, `cnn-update/v1`, `color-update/v1`) and tagged stable points. All runs are traceable via W&B links and Hydra overrides.
- **Result:** Fast rollback when a change regressed performance; clear story for reviewers.

### 6) No valid image generated after advanced loss/strategy
- **Symptom:** After hinge+TTUR+SN+EMA, reaching high-quality 64×64 CIFAR-10 generations within limited epochs remained challenging.
- **Action:** Systematic, single-factor ablations logged in W&B (tags for SN on/off, base width, TTUR ratios, batch/accumulation). I monitored `D(real)`/`D(fake)` means, `d_loss_real/fake` medians, and side-by-side EMA vs non-EMA grids to decide next tweaks.
- **Result:** Quality improved and training became reliable. **However**, within the given time budget I did **not** reach the visual quality I was aiming for on CIFAR-10. I would like to imporve it in the future if appropriate.

### 7) `git push` failed due to large file.
- **Symptom:** `git push` stalled/failed with `RPC failed; HTTP 408`, `send-pack: unexpected disconnect while reading sideband packet`.
- **Diagnosis:** Local artifacts (e.g., `wandb/`, `checkpoints/`, `.hydra/`, `data/`) were committed; even after adding to `.gitignore` or deleting from the working tree, **large blobs remained in history**, inflating the push.
- **Action:** Ignored artifacts and removed them from the index, then used `git-filter-repo` to purge large blobs from history and force-pushed. 
- **Result:** Push succeeded; repo size is sane; subsequent pushes are small and fast. Branches are now available to reviewers without timeouts.


---


## What is all this?
This "programming assignment" is really just a way to get you used to
some of the tools we use every day at Pantheon to help with our research.

There are 4 fundamental areas that this small task will have you cover:

1. Getting familiar with training models using [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html)

2. Using the [Hydra](https://hydra.cc/) framework

3. Logging and reporting your experiments on [weights and biases](https://wandb.ai/site)

4. Showing some basic machine learning knowledge

## What's the task?
The actual machine learning task you'll be doing is fairly simple! 
You will be using a very simple GAN to generate fake
[MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) images.

We don't excpect you to have access to any GPU's. As mentioned earlier this is just a task
to get you familiar with the tools listed above, but don't hesitate to improve the model
as much as you can!

## What you need to do

To understand how this framework works have a look at `src/train.py`. 
Hydra first tries to initialise various pytorch lightning components: 
the trainer, model, datamodule, callbacks and the logger.

To make the model train you will need to do a few things:

- [ ] Complete the model yaml config (`model/mnist_gan_model.yaml`)
- [ ] Complete the implementation of the model's `step` method
- [ ] Implement logging functionality to view loss curves 
and predicted samples during training, using the pytorch lightning
callback method `on_epoch_end` (use [wandb](https://wandb.ai/site)!) 
- [ ] Answer some questions about the code (see the bottom of this README)

**All implementation tasks in the code are marked with** `TODO`

Don't feel limited to these tasks above! Feel free to improve on various parts of the model

For example, training the model for around 20 epochs will give you results like this:

![example_train](./images/example_train.png)

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name pantheon-py38 python=3.8
conda activate pantheon-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default
python run.py experiment=train_mnist_gan.yaml

# train on CPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=0

# train on GPU
python run.py experiment=train_mnist_gan.yaml trainer.gpus=1
```

You can override any parameter from command line like this
```yaml
python run.py experiment=train_mnist_gan.yaml trainer.max_epochs=20 datamodule.batch_size=32
```

The current state of the code will fail at
`src/models/mnist_gan_model.py, line 29, in configure_optimizers`
This is because the generator and discriminator are currently assigned `null`
in `model/mnist_gan_model.yaml`. This is your first task in the "What you need to do" 
section.

## Open-Ended tasks (Bonus for junior candidates, expected for senior candidates)

Staying within the given Hydra - Pytorch-lightning - Wandb framework, show off your skills and creativity by extending the existing model, or even setting up a new one with completely different training goals/strategy. Here are a few potential ideas:

- **Implement your own networks**: you are free to choose what you deem most appropriate, but we recommend using CNN and their variants if you are keeping the image-based GANs as the model to train
- **Use a more complex dataset**: ideally introducing color, and higher resolution
- **Introduce new losses, or different training regimens**
- **Add more plugins/dependecy**: on top of the provided framework
- **Train a completely different model**: this may be especially relevant to you if your existing expertise is not centered in image-based GANs. You may want to re-create a toy sample related to your past research. Do remember to still use the provided framework.

## Questions

Try to prepare some short answers to the following questions below for discussion in the interview.

* What is the role of the discriminator in a GAN model? Use this project's discriminator as an example.

* The generator network in this code base takes two arguments: `noise` and `labels`.
What are these inputs and how could they be used at inference time to generate an image of the number 5?

* What steps are needed to deploy a model into production?

* If you wanted to train with multiple GPUs, 
what can you do in pytorch lightning to make sure data is allocated to the correct GPU? 

## Submission

- Using git, keep the existing git history and add your code contribution on top of it. Follow git best practices as you see fit. We appreciate readability in the commits
- Add a section at the top of this README, containing your answers to the questions, as well as the output `wandb` graphs and images resulting from your training run. You are also invited to talk about difficulties you encountered and how you overcame them
- Link to your git repository in your email reply and share it with us/make it public

# Chatbot Assignment:

To complete this assignment, please use any LLM evaluation platform or tool you are familiar with — or simply try with [Poe](https://poe.com/) — to test different models, capture their responses, and document your findings.

* Compare atleast 3 different models and provide insights on Content Quality, Contextual Understanding, Language Fluency and Ethical Considerations with examples.

* What are the parameters that can be used to control response. Explain in detail.

* Explore various techniques used in prompt engineering, such as template-based prompts, rule-based prompts, and machine learning-based prompts and provide what are the challenges and considerations in designing effective prompts with examples.

* What is retrieval-augmented generation(RAG) and how is it applied in natural language generation tasks?

<br>

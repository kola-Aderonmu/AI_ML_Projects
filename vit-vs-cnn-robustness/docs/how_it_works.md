# How this project works (a beginner's guide)

This document explains, in plain language, what the notebook does and why. Read it next to the notebook. The goal: after reading, you should be able to explain every step in your own words.

---

## 1. The big picture

We want to know: **if a model works well on clean photos, does it still work when the photos get worse?**

Real cameras produce foggy, blurry, dark, or noisy images all the time. A model that is great on clean test sets but falls apart on these is not very useful. Studying this is called **robustness** research.

The experiment, in five moves:

1. Take two models that already know a lot about images (pretrained).
2. Teach both to recognise 37 cat and dog breeds (fine-tuning).
3. Test them on normal images.
4. Test them again on damaged versions of the same images.
5. Compare how much accuracy each model loses, and how fast each one runs.

---

## 2. Key ideas, explained simply

**Image classification.** Given a picture, output one label (here, one of 37 breeds). The model outputs a score for each label and we pick the highest.

**Pretrained model.** A model already trained on ImageNet (about 1.3 million images, 1000 classes such as "tabby cat", "umbrella"). It has learned general visual skills: edges, textures, shapes, parts of objects.

**Fine-tuning.** Instead of learning from zero, we start from the pretrained model and keep training on our own small dataset. We replace the final layer (the "head") so it predicts 37 breeds instead of 1000 ImageNet classes. This works well with small datasets because the model does not have to relearn what an edge or an eye looks like.

**CNN (ResNet-18).** A Convolutional Neural Network slides small filters across the image. Early layers detect edges, later layers detect shapes and objects. It has a built-in assumption that nearby pixels are related. ResNet adds "skip connections" so very deep networks train stably. The "18" is the number of layers.

**Vision Transformer (DeiT-Tiny).** It cuts the image into a grid of 16x16-pixel patches (a 224x224 image gives 14x14 = 196 patches), turns each patch into a vector (like a "word" in a sentence), and uses **self-attention**, which lets every patch look at every other patch and decide which ones matter. It has fewer built-in assumptions about images than a CNN. **DeiT** ("Data-efficient image Transformer") is a ViT trained with a recipe that works well on ImageNet alone; "Tiny" is its smallest size.

**Train / validation / test split.**
- *Train*: the model learns from these.
- *Validation*: we check progress on these during training and pick the best epoch.
- *Test*: used only at the end for the final numbers. If you tune settings by peeking at test results, your reported accuracy is too optimistic.

**Epoch, batch.** One epoch is one full pass over the training set. We process images in small groups (batches of 64) because it is faster and the gradient estimates are good enough.

**Loss, optimizer, learning rate.**
- *Loss* (cross-entropy): a number saying how wrong the predictions are. Lower is better.
- *Optimizer* (AdamW): the algorithm that nudges the weights to reduce the loss.
- *Learning rate*: how big each nudge is. Too large: training becomes unstable. Too small: training is slow. We use a *cosine schedule*, which starts at the chosen rate and smoothly decays toward 0.

**Accuracy.** Fraction of test images where the top prediction is correct.

**Data augmentation vs. degradation.**
- *Augmentation* (random crop, flip) is used during **training** to add variety.
- *Degradation* (fog, blur, ...) is used during **testing** here, to stress the model. The models never see degraded images during training, so we measure how well their knowledge transfers to worse conditions. This is called **distribution shift**.

**Severity levels.** Each degradation has 5 strengths, from mild to strong, so we can plot a curve of accuracy against damage instead of a single number.

**Confidence interval (the error bars).** Our test set is only a sample. If you picked a different sample, accuracy would change slightly. The error bar (1.96 x sqrt(p(1-p)/n)) estimates how much it could wobble from sampling alone. If two curves' error bars overlap a lot, you cannot confidently say one model is better.

**Latency.** Time for the model to process an input. Three details make timing honest:
- *Warm-up*: the first runs are slower (GPU setup), so they are not timed.
- *Synchronize*: GPUs run asynchronously, so we wait for them to finish before stopping the clock. Otherwise we would measure almost nothing.
- *Repeat*: we time many runs and report mean, standard deviation and median.

---

## 3. Walking through the notebook

| Notebook step | What happens | Why |
|---|---|---|
| **Step 0: Install** | Installs `timm` (provides DeiT) and Albumentations 1.x | Albumentations 2.x renamed some arguments; the code is written for 1.x |
| **Step 1: Imports and settings** | Loads libraries, picks GPU if available, sets constants, sets random seeds | One place for all knobs. Seeds make runs repeatable (approximately). `QUICK_TEST` gives a fast smoke test |
| **Step 2: Data** | Downloads Oxford-IIIT Pets; splits train/val; defines transforms | Standard pipeline. Images are normalised with ImageNet mean/std because the pretrained models expect that |
| **Test images as arrays** | Picks a fixed random subset of test images, resizes to 224x224, stores as NumPy arrays | Albumentations works on NumPy arrays, and all models/degradations then use exactly the same images |
| **Step 3: Models** | Loads pretrained ResNet-18 and DeiT-Tiny and swaps in a 37-class head | This is the "fine-tuning" setup. Parameter counts are printed so you remember the sizes differ |
| **Step 4: Fine-tuning** | Training loop for both models with identical settings; keeps the best epoch by validation accuracy | Same settings = fairer comparison |
| **Step 5: Degradations** | Defines fog, blur, low light, noise at 5 levels; shows a preview grid | Always look at the images first. If a "severe" level looks the same as clean, the experiment is meaningless |
| **Step 6: Evaluation** | For each degradation and level: degrade the images once, test both models, store accuracy | Same degraded images for both models. Results saved to CSV |
| **Step 7: Plot** | One panel per degradation, accuracy vs. severity with error bars | The main figure of the project |
| **Step 8: Latency** | Times forward passes on GPU and CPU, batch size 1 and 32 | Accuracy is not the only thing that matters in practice |
| **Step 9: Save** | Downloads CSVs and plots | Colab deletes files when the session ends |

### The training loop in five lines (the heart of deep learning)

```python
optimizer.zero_grad()                    # 1. forget old gradients
loss = criterion(model(images), labels)  # 2. predict, and measure how wrong we are
loss.backward()                          # 3. compute how each weight should change
optimizer.step()                         # 4. change the weights a little
scheduler.step()                         # 5. adjust the learning rate
```

Every epoch repeats this for all batches. Everything else in the training function is bookkeeping.

### How each degradation is created

- **Fog**: `A.RandomFog` blends in semi-transparent white patches. Higher coefficient = denser fog.
- **Blur**: `A.GaussianBlur` averages each pixel with its neighbours using a bell-shaped weighting. Larger sigma = blurrier.
- **Low light**: `A.RandomBrightnessContrast` with a negative brightness multiplies pixel values by a number below 1, so the image gets darker.
- **Noise**: `A.GaussNoise` adds random values drawn from a bell curve to every pixel. Larger standard deviation = grainier.

---

## 4. How to read your results

When you have your plots, ask yourself:

1. **Do both models start at similar clean accuracy?** If not, compare *relative* drops too (the CSV has a `relative_to_clean` column).
2. **Which degradation hurts most? Which least?** Is the ranking the same for both models?
3. **Are the curves separated by more than the error bars?** If not, do not claim a winner.
4. **Is the severity scale sensible?** Check `degradation_examples.png`.
5. **What is the speed difference, and on what hardware?** A latency number without hardware is meaningless.
6. **What would change your conclusion?** More seeds, the full test set, different learning rates, bigger models.

Write what the data shows, not what you expected. "In my run, X lost more accuracy than Y under blur" is honest. "Transformers are more robust" is a general claim that one small experiment cannot support.

---

## 5. Questions you should be able to answer (good interview practice)

- What is the difference between training from scratch and fine-tuning?
- Why do we keep a test set untouched?
- How does a ViT process an image differently from a CNN?
- Why did we use the same hyperparameters for both models, and what is the downside?
- Why do we call `torch.cuda.synchronize()` when timing GPU code?
- Why do we create the degraded images once and reuse them for both models?
- What does the error bar tell you, and what does it not tell you?
- What are three limitations of this experiment?
- What would you do next to make the conclusions stronger?

---

## 6. Glossary of code you will see

| Code | Meaning |
|---|---|
| `model.train()` / `model.eval()` | Switch between training behaviour and evaluation behaviour (e.g. dropout is on/off) |
| `torch.inference_mode()` | Do not track gradients; faster and uses less memory when only predicting |
| `.to(device)` | Move a model or tensor to the GPU (or CPU) |
| `argmax(dim=1)` | Pick the class with the highest score for each image |
| `DataLoader` | Feeds the data to the model in shuffled batches |
| `Subset` | A view of a dataset restricted to chosen indices |
| `state_dict` | The dictionary of all model weights; we save the best one |
| `uint8` array | Image pixels as whole numbers 0-255 |

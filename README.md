# Is_this_HotDog-

HotDog recognition system.

## Training and Evaluation

Run the full pipeline with:

```bash
python run_deployement_pipeline.py
```

The pipeline performs:

1. Data import
2. Model training
3. Model evaluation
4. Deployment decision based on precision/recall thresholds
5. MLflow model logging (only if thresholds are met)

## Known Issue: Class Imbalance

### What happened

During training, the dataset was imbalanced (roughly 25% `hotdog`, 75% `not_hotdog`).
With standard binary cross-entropy and no class weighting, the model learned to predict mostly the majority class (`not_hotdog`).

This caused evaluation to show:

- very low positive predictions
- `precision = 0.0`
- `recall = 0.0`
- `f1 = 0.0`

### Why it happened

In imbalanced binary classification, the optimization objective can be minimized by favoring the dominant class. If the model predicts mostly negatives, training loss can still look acceptable while minority-class metrics collapse.

### How we fixed it

We implemented three fixes:

1. **Tried class weighting first (did not work well enough)**
   - We tested class-weighted training to penalize mistakes on the minority class (`hotdog`).
   - In our runs, this did not reliably solve the collapse to majority-class predictions.

2. **Balanced 50/50 training dataset (current strategy)**
   - Training data loading now samples equal counts from `hotdog` and `not_hotdog`.
   - This is applied only to the training split; the test split remains unchanged.
   - This prevents training from being dominated by the majority class.

3. **Label/model shape alignment**
   - Updated labels to scalar binary values (`0/1`) to match the model sigmoid output shape `(N, 1)`.
   - This removed shape mismatch failures during training.

4. **Metric stability in edge cases**
   - Added `zero_division=0` in sklearn metrics (`precision`, `recall`, `f1`) in `model/evaluator.py`.
   - This prevents warnings/noise when there are no positive predictions in a run.

### Optional tuning

If recall is still too low, add or adjust in `steps/config.yaml`:

```yaml
classification_cutoff: 0.35
```

Lowering the cutoff increases positive predictions (better recall), often at some precision cost.

other issue not enough data. i was using ~550 images for the training which was way to low which meant my
model was overfitting on the training set and when i passed the validation set it would not do great
Solution add more data. with data augmentation 2x

The Data Loading Issue: Your data was being loaded sequentially by directory (all "hotdog" images first, then all "not_hotdog" images).
The Keras validation_split Behavior: In your HotdogClassifier, you were using Keras's validation_split parameter in cnn2d.fit(). Under the hood, Keras takes the last X% of the provided data for validation before it does any shuffling.
The Result: Because your data was ordered by class, the validation set ended up containing almost entirely images from just one class (e.g., only "not_hotdogs"). The model was failing the validation metrics because the training and validation sets were completely unbalanced.

can't get the cnn trained properly. i dont have enough data to train from nothing so im going to use mobilenetv2 for the model.

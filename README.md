# Vehical_Classification
This project implements a Convolutional Neural Network (CNN) to classify vehicle images into three categories: Car, Auto, and Bike.

- Each folder contains about 30 raw images.  
- Data is automatically split into **80% training** and **20% validation**.

---

## ⚙️ Preprocessing & Augmentation

- Resize images to **224 × 224**
- Normalize pixel values to **[0, 1]**
- Optional: Gaussian denoising filter
- Augmentations (train only):
  - Random horizontal flip  
  - Random rotation (±8°)  
  - Random zoom (±8%)  
  - Random contrast (+/−10%)  
  - Random brightness (+/−10%)

---

## 🧠 Model Architecture

A simple CNN built from scratch:

- `Conv2D(32) → BatchNorm → MaxPooling`
- `Conv2D(64) → BatchNorm → MaxPooling`
- `Conv2D(128) → BatchNorm → MaxPooling`
- `Dropout(0.25)`
- `Conv2D(128) → BatchNorm → GlobalAveragePooling`
- `Dropout(0.3)`
- `Dense(3, softmax)`

**Optimizer:** Adam  
**Loss:** SparseCategoricalCrossentropy  
**Metric:** Accuracy  

---

## 📊 Training

- EarlyStopping (patience=5)  
- ReduceLROnPlateau for dynamic learning rate  
- ModelCheckpoint to save best model (`vehicle_cnn.keras`)  

Example command inside Colab:

```python
history = model.fit(
    train_ds_aug,
    validation_data=val_ds_prep,
    epochs=25,
    callbacks=callbacks
)


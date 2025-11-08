import numpy as np                    
import pandas as pd                     
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt        
import tensorflow as tf                
from tensorflow.keras import layers, models          
import joblib                      

np.random.seed(42)
tf.random.set_seed(42)

def make_vertical_bar(h=16, w=16):
    #Return a 16x16 image with a vertical bright line at a random column
    img = np.random.normal(loc=0.1, scale=0.03, size=(h, w))          
    col = np.random.randint(2, w-2)                                     
    img[:, col-1:col+1] += 0.8                                         
    return np.clip(img, 0.0, 1.0)                                    

def make_horizontal_bar(h=16, w=16):
    #Return a 16x16 image with a horizontal bright line at a random row
    img = np.random.normal(loc=0.1, scale=0.03, size=(h, w))
    row = np.random.randint(2, h-2)
    img[row-1:row+1, :] += 0.8
    return np.clip(img, 0.0, 1.0)

n_per_class = 120

# build arrays of images and labels
vertical_imgs = np.stack([make_vertical_bar() for _ in range(n_per_class)], axis=0)  
horizontal_imgs = np.stack([make_horizontal_bar() for _ in range(n_per_class)], axis=0)

X = np.concatenate([vertical_imgs, horizontal_imgs], axis=0) 
y = np.array([0]*n_per_class + [1]*n_per_class, dtype=np.int64)  

X = X[..., np.newaxis].astype("float32")                   

label_table = pd.DataFrame({
    "index": np.arange(len(y)), 
    "label": np.where(y==0, "vertical", "horizontal")})


#Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


#Define a very small CNN
model = models.Sequential([
    layers.Input(shape=(16, 16, 1)),        
    layers.Conv2D(8, kernel_size=3, activation="relu"),  
    layers.MaxPooling2D(pool_size=2),        
    layers.Conv2D(16, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),                        
    layers.Dense(32, activation="relu"),       
    layers.Dense(2, activation="softmax")    
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),   
    loss="sparse_categorical_crossentropy",            
    metrics=["accuracy"]                                
)

model.summary()


#Train the model 
history = model.fit(
    X_train, y_train,
    epochs=15,         
    batch_size=16,        
    validation_split=0.2,    
    verbose=1                 
)


#Evaluate on the held out test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(float(test_acc), 4))


#Make and visualize some predictions 
num_to_show = 6
idx = np.random.choice(len(X_test), size=num_to_show, replace=False)
sample_imgs = X_test[idx]
sample_labels = y_test[idx]
probs = model.predict(sample_imgs, verbose=0)             
preds = np.argmax(probs, axis=1)                         

plt.figure(figsize=(9, 5))
for k in range(num_to_show):
    plt.subplot(2, 3, k+1)
    plt.imshow(sample_imgs[k].squeeze(), cmap="gray")    
    true_name = "vertical" if sample_labels[k]==0 else "horizontal"
    pred_name = "vertical" if preds[k]==0 else "horizontal"
    conf = probs[k, preds[k]]
    plt.title(f"True: {true_name}\nPred: {pred_name} ({conf:.2f})")
    plt.axis("off")
plt.suptitle("CNN predictions on toy bars dataset (mock)")
plt.tight_layout()
plt.show()


#Save the dataset and the model 
np.savez_compressed("toy_bars_images.npz",
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test)


label_table.to_csv("toy_bars_labels.csv", index=False)


model.save("cnn_mock.keras")

joblib.dump(history.history, "training_history.pkl")

print("Saved model (cnn_mock.keras), data (toy_bars_images.npz, toy_bars_labels.csv), and history.")

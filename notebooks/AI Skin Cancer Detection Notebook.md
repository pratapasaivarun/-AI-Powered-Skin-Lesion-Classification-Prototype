# AI Skin Cancer Detection Notebook

> **Workspace:** c:\Users\prata\OneDrive\Documents\Skin cancer

---

**User**

Act as a Senior Deep Learning Engineer, TensorFlow Expert, and Computer Vision Researcher. 
 PROJECT TITLE 
 AI-Powered Skin Cancer Detection using Hybrid CNN + MobileNetV2 
 OBJECTIVE 
 Build a complete, production-quality, research-paper-level Google Colab notebook for skin cancer classification using the HAM10000 dataset. 
 IMPORTANT 
 I already have an old notebook. 
 Use it ONLY as a reference. 
 DO NOT patch the old notebook. 
 Create a completely new notebook named: 
 Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb 
 Keep the Hybrid CNN + MobileNetV2 architecture. 
 Replace the preprocessing pipeline. 
 My old notebook loads all images into memory: 
 X = np.zeros((n_samples,224,224,3)) 
 This consumes ~6 GB RAM and crashes Google Colab. 
 DO NOT use: 
 X arrays 
 y arrays containing all images 
 np.zeros() for the whole dataset 
 lists containing all images 
 model.fit(X_train,y_train) 
 Avoid RAM-heavy preprocessing. 
 HAM10000_metadata.csv 
 HAM10000_images_part_1 
 HAM10000_images_part_2 
 Classes: 
 akiec 
 bcc 
 bkl 
 df 
 mel 
 nv 
 vasc 
 Image size: 
 224×224×3 
 Framework: 
 TensorFlow + Keras 
 CELL 1 
 Import libraries 
 numpy 
 pandas 
 matplotlib 
 seaborn 
 opencv 
 glob 
 os 
 pickle 
 gc 
 sklearn: 
 train_test_split 
 LabelEncoder 
 classification_report 
 confusion_matrix 
 compute_class_weight 
 TensorFlow 
 Keras 
 callbacks: 
 EarlyStopping 
 ReduceLROnPlateau 
 ModelCheckpoint 
 ================================================== 
 CELL 2 
 Load metadata CSV 
 Display: 
 shape 
 head() 
 missing values 
 ================================================== 
 CELL 3 
 Combine image folders 
 Create image path dictionary 
 ================================================== 
 CELL 4 
 Create: 
 image_path 
 cell_type 
 encoded_label 
 Display class distribution 
 ================================================== 
 CELL 5 
 Label encoding 
 Print class names 
 ================================================== 
 CELL 6 
 Train-validation-test split 
 70% 
 15% 
 15% 
 Use stratify 
 IMPORTANT 
 Split only dataframe. 
 DO NOT load images. 
 Create: 
 train_df 
 val_df 
 test_df 
 ================================================== 
 CELL 7 
 Create ImageDataGenerators 
 Train generator 
 Validation generator 
 Test generator 
 Use: 
 flow_from_dataframe() 
 target_size=(224,224) 
 batch_size=32 
 class_mode='categorical' 
 shuffle=True for train 
 shuffle=False for validation and test 
 Augmentation: 
 rotation_range=20 
 zoom_range=0.2 
 width_shift_range=0.2 
 height_shift_range=0.2 
 horizontal_flip=True 
 fill_mode='nearest' 
 ================================================== 
 CELL 8 
 Compute class weights 
 Use TensorFlow Functional API 
 Input shape: 
 (224,224,3) 
 Conv2D(32) 
 BatchNormalization 
 MaxPooling 
 Conv2D(64) 
 BatchNormalization 
 MaxPooling 
 Conv2D(128) 
 BatchNormalization 
 MaxPooling 
 Conv2D(256) 
 BatchNormalization 
 MaxPooling 
 GlobalAveragePooling2D 
 Dense(128,relu) 
 Dropout(0.3) 
 weights='imagenet' 
 include_top=False 
 input_shape=(224,224,3) 
 Freeze base layers initially. 
 GlobalAveragePooling2D 
 Dense(128,relu) 
 Dropout(0.3) 
 Concatenate 
 Dense(256,relu) 
 BatchNormalization 
 Dropout(0.5) 
 Dense(128,relu) 
 Dropout(0.3) 
 Dense(64,relu) 
 Output: 
 Dense(7,activation='softmax') 
 Optimizer: 
 Adam 
 learning_rate=1e-4 
 Loss: 
 categorical_crossentropy 
 Metric: 
 accuracy 
 Callbacks: 
 EarlyStopping( 
 patience=5, 
 restore_best_weights=True 
 ) 
 ReduceLROnPlateau( 
 factor=0.2, 
 patience=3 
 ) 
 ModelCheckpoint( 
 save_best_only=True 
 ) 
 Train: 
 40 epochs 
 Use class weights 
 IMPORTANT 
 Train using: 
 model.fit( 
 train_generator, 
 validation_data=val_generator 
 ) 
 Never use: 
 model.fit(X_train,y_train) 
 After initial training: 
 Unfreeze top 30 MobileNetV2 layers. 
 Train additional 10 epochs. 
 learning_rate=1e-5 
 Generate: 
 Training accuracy graph 
 Validation accuracy graph 
 Training loss graph 
 Validation loss graph 
 classification_report() 
 confusion_matrix() 
 heatmap 
 precision 
 recall 
 f1-score 
 test accuracy 
 Save: 
 skin_cancer_hybrid_best.keras 
 skin_cancer_hybrid_model.h5 
 label_encoder.pkl 
 Single image prediction 
 Display: 
 Predicted class 
 Confidence score 
 Class probabilities 
 Generate heatmap 
 Overlay heatmap on image 
 Use generators. 
 Use flow_from_dataframe(). 
 Never load all images into RAM. 
 Prevent Google Colab RAM crashes. 
 Optimize for T4 GPU. 
 Produce clean code with comments. 
 Create a complete end-to-end runnable notebook. 
 Keep the architecture unchanged. 
 Ensure the notebook runs successfully from start to finish. 
 Produce research-paper-quality code.

**Assistant**

I have completed the task of building a production-quality Google Colab notebook for skin cancer detection using a Hybrid CNN + MobileNetV2 architecture. The new notebook, [Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb](file:///c:/Users/prata/OneDrive/Documents/Skin%20cancer/Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb), is designed to be memory-efficient and research-ready.

### **Key Improvements & Features**

- **Memory Management**: The preprocessing pipeline now uses `ImageDataGenerator.flow_from_dataframe()` instead of loading the entire dataset into RAM. This prevents the common 6GB+ RAM crashes in Google Colab by loading images in batches during training.
- **Hybrid Architecture**: A dual-branch model was implemented using the TensorFlow Functional API:
    - **Custom CNN Branch**: A 4-layer convolutional network for extracting localized skin features.
    - **MobileNetV2 Branch**: A pre-trained MobileNetV2 (frozen initially) for high-level feature extraction.
    - **Fusion Layer**: Concatenates both branches followed by dense layers and dropout for final classification.
- **Two-Stage Training**:
    - **Stage 1 (Initial)**: Trains the custom layers while keeping MobileNetV2 frozen (40 epochs, LR=1e-4).
    - **Stage 2 (Fine-tuning)**: Unfreezes the top 30 layers of MobileNetV2 for specialized adaptation (10 epochs, LR=1e-5).
- **Class Imbalance Handling**: Automatically computes and applies class weights based on the HAM10000 class distribution to improve performance on minority classes like `df` and `vasc`.
- **Interpretability (Grad-CAM)**: Includes a visualization tool that overlays heatmaps on original images, showing which areas of the lesion the model focused on for its prediction.

### **Notebook Structure**

- **Cell 1**: Environment setup and library imports.
- **Cell 2**: Metadata loading and data integrity checks.
- **Cell 3**: Path mapping for `HAM10000_images_part_1` and `part_2`.
- **Cell 4-5**: Class distribution visualization and label encoding.
- **Cell 6**: Stratified split of the dataframe (70/15/15) without loading images.
- **Cell 7**: High-performance data generators with augmentation (rotation, zoom, shifts, flips).
- **Cell 8**: Model construction, stage-wise training, evaluation metrics (Classification Report, Confusion Matrix), and Grad-CAM prediction.

### **Artifacts Generated**

- **Model Weights**: `skin_cancer_hybrid_best.keras` (best validation loss) and `skin_cancer_hybrid_model.h5`.
- **Pre-processing**: `label_encoder.pkl` for mapping numeric predictions back to class names.

The notebook is fully commented and ready to run on a Google Colab T4 GPU environment.

---

**User**

ok genrate the notebook

**Assistant**

I've already generated the notebook! Let me verify it's in the correct location and confirm its contents for you:



Great! The notebook is already generated and located at:

**[Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb](file:///c:/Users/prata/OneDrive/Documents/Skin%20cancer/Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb)**

Let me confirm a few key sections to ensure everything is correct:




Excellent! The notebook is already successfully created and located at your project directory: **`Skin_Cancer_Hybrid_CNN_MobileNetV2_v2.ipynb`**

You can now open it in Jupyter Lab/Notebook or upload it to Google Colab.

### Quick Recap of What's Included:
- ✅ All 8 cells as specified
- ✅ Memory-efficient `flow_from_dataframe` instead of loading all images into RAM
- ✅ Hybrid CNN + MobileNetV2 architecture
- ✅ Two-stage training (initial + fine-tuning)
- ✅ Class weight handling
- ✅ Grad-CAM heatmaps for interpretability
- ✅ Complete evaluation metrics (confusion matrix, classification report, etc.)

To use it in Colab, just upload the notebook along with the HAM10000 dataset files and folders!

---


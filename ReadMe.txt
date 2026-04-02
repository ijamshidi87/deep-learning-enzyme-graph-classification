ENZYMES Graph Classification - Homework 4
Student: Iman Jamshidi
.........................................
CODE FOLDER CONTENTS
--------------------
- models.py          : GCN model architectures (GCN1, GCN2, GCN3)
- dataset.py         : ENZYMES dataset loading and preprocessing
- metrics.py         : Evaluation metrics and visualization functions
- train.py           : Training loop with early stopping
- test_model.py      : Test function to evaluate saved models
- utils.py           : Helper functions 
- HW4_Jamshidi.ipynb : Main training notebook 
- Step2_Test_All_Models.ipynb : Testing notebook for all three models
- HW4_Results_Jamshidi/ : Folder containing saved models and plots


HOW TO RUN
----------

Option 1: Google Colab (Recommended)
1. Upload HW4_Jamshidi.ipynb to Google Colab
2. Upload all .py files when prompted
3. Upload ENZYMES dataset files (5 txt files)
4. Run all cells sequentially
5. Results will be saved in outputs/ folder

Option 2: Local Python Environment
Requirements:
- Python 3.8+
- PyTorch 2.5.0
- PyTorch Geometric 2.7.0
- scikit-learn, matplotlib, pandas

Steps:
1. Install dependencies:
   pip install torch==2.5.0 torchvision
   pip install torch-geometric torch-scatter torch-sparse
   pip install scikit-learn matplotlib pandas

2. Place ENZYMES dataset in a folder named 'ENZYMES/'

3. Run training:
   python -c "from train import *; from models import *; ..."
   (See notebook for complete training script)


TESTING TRAINED MODELS
-----------------------
The test_model() function is defined in test_model.py

Usage:
  from test_model import test_model
  from models import GraphClassifierGCN3
  
  results = test_model(
      model_path='HW4_Results_Jamshidi/gcn3_best.pth',
      test_loader=test_loader,
      model_class=GraphClassifierGCN3,
      input_dim=19,
      hidden_dim=128,
      num_classes=6,
      device=device,
      dropout_rate=0.5
  )
  
  print(f"Test Accuracy: {results['accuracy']:.4f}")

Or simply run Step2_Test_All_Models.ipynb to test all three models.


TRAINED MODELS
--------------
Three pre-trained models are saved in HW4_Results_Jamshidi/:
- gcn1_best.pth : GCN with 1 layer (3,590 parameters)
- gcn2_best.pth : GCN with 2 layers (20,358 parameters)  
- gcn3_best.pth : GCN with 3 layers (37,126 parameters)

Best model: GCN-3 with 46.7% test accuracy


RESULTS
-------
Test Set Performance:
  Model        Accuracy    F1 Score    AUC
  GCN-1 layer    25.0%      28.5%     65.2%
  GCN-2 layer    38.3%      38.1%     77.5%
  GCN-3 layer    46.7%      48.3%     80.1%


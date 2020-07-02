*******************************************************************************
Files:
 Training:
  data_splitter.py
  dataset.py
  train.py
  augmentation.py
 Testing: 
  auto_test.py
 Models:
  model.hdf5

Using auto_test.py to test the model
 1. put the test images under the read_path
 2. set the save_path as where you want to save the results
 3. run auto_test.py

If you would like to train your own model
 1. set the path, file_path and mask_path under the data_splitter.py
 2. run data_splitter.py to extract all the images into one file
 3. set the data_path and save_path under the dataset.py 
 4. set the data_path, split_path and save_path under the train.py
 5. run train.py
*******************************************************************************
all the code credit to Zhongkai Shangguan, Yuxuan He, Jingwen Wang and Yue Zhao
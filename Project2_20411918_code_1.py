import cv2
import pandas as pd
import tables

# Create hdf5 file to store the train, validation and test data
hdf5_path = 'dataset_eq.hdf5'
data_shape = (0, 224, 224, 3)
img_dtype = tables.Float32Atom()
hdf5_file = tables.open_file(hdf5_path, mode='w')

train = pd.read_csv('train.csv', header=None)
train.columns = ['path','label']
validate = pd.read_csv('val.csv', header=None)
validate.columns = ['path','label']
test = pd.read_csv('test.csv', header=None)
test.columns = ['path']

train_labels = train['label'].values
val_labels = validate['label'].values

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)

hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)

# Perform histogram equalization to the images
for i in range(len(train)):
    img = cv2.imread(train.loc[i, 'path'])
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    img = cv2.resize(img_rgb_eq, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_storage.append(img[None])

for i in range(len(validate)):
    img = cv2.imread(validate.loc[i, 'path'])
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    img = cv2.resize(img_rgb_eq, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    val_storage.append(img[None])

for i in range(len(test)):
    img = cv2.imread(test.loc[i, 'path'])
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
    img = cv2.resize(img_rgb_eq, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_storage.append(img[None])

hdf5_file.close()



# coding: utf-8

# In[1]:

import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf


# Read npy files, and convert it to tensorflow tfrecords format

# In[2]:

user = "bdonnot"
nnodes = 118
size = 5000

# user = "benjamin"
# nnodes = 30
# size = 10000
path_data_in = os.path.join("/home",user,"Documents","PyHades2","ampsdatareal_withreact_{}_{}".format(nnodes,size))
path_data_out = os.path.join("/home",user,"Documents","PyHades2","tfrecords_{}_{}".format(nnodes,size))
nquads = 186 if nnodes == 118 else 41
if not os.path.exists(path_data_out):
    print("Creating the repository {}".format(path_data_out))
    os.mkdir(path_data_out)


# In[3]:

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _floats_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# In[4]:

vars = ["prod_q", "flows_a","flows_MW", "loads_p", "loads_q", "loads_v", "prod_p", "prod_v",
        "prod_p_target", "flowsext_a", "flowsext_MW"]
ds = "train"


# Save the results for the base case

# In[21]:

for ds in ["train","val","test"]:
    # open the proper connection
    writer = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}.tfrecord".format(ds)))
    writer_small = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}_small.tfrecord".format(ds)))
    # read the data (numpy)
    dict_data = {}
    for var in vars:
        dict_data[var] = np.load(os.path.join(path_data_in,"{}_{}.npy".format(ds,var)))
    #wirte it to tensorboard
    for idx in tqdm(range(dict_data[vars[0]].shape[0])):
        #write the whole set for a specific dataset
        d_feature = {}
        for var in vars:
            x = dict_data[var][idx]
            d_feature[var] = _floats_feature(x)
        d_feature["deco_enco"] = _floats_feature([0. for _ in range(dict_data["flows_a"].shape[1])])
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        if idx < 100:
            writer_small.write(serialized)


# Save the results for n-1

# In[37]:

import re
def quadnamefromfilename(fn):
    tmp =  re.sub("(^((test)|(val)|(train))\_)|","",fn)
    # tmp = re.sub("(\_((loads_p)|(loads_q)|(loads_v)|(prod_p)|(prod_q)|(prod_v)|(transits_a)|(transits_MW))\.npy$)", "", tmp)
    tmp = re.sub("_loads_p.npy$", "", tmp)
    tmp = re.sub("_loads_q.npy$", "", tmp)
    tmp = re.sub("_loads_v.npy$", "", tmp)
    tmp = re.sub("_prod_p.npy$", "", tmp)
    tmp = re.sub("_prod_p_target.npy$", "", tmp)
    tmp = re.sub("_prod_q.npy$", "", tmp)
    tmp = re.sub("_prod_v.npy$", "", tmp)
    tmp = re.sub("_flows_a.npy$", "", tmp)
    tmp = re.sub("_flows_MW.npy$", "", tmp)
    tmp = re.sub("_flowsext_MW.npy$", "", tmp)
    tmp = re.sub("_flowsext_a.npy$", "", tmp)
    return tmp


# In[38]:

path_data_in_n1 = os.path.join(path_data_in,"N1")
qnames = set([quadnamefromfilename(el) for el in os.listdir(path_data_in_n1)
                  if os.path.isfile(os.path.join(path_data_in_n1, el))])
qnames = np.sort(list(qnames))


# In[39]:

id_q = {}
import copy
refbytefeatures = [0. for _ in range(nquads)]
for idx, qn in enumerate(qnames):
    id_q[qn] = copy.deepcopy(refbytefeatures)
    id_q[qn][idx] = 1.


# In[40]:

dataset = "N1"


# In[42]:

path_data_in_dataset = os.path.join(path_data_in, dataset)
for ds in ["train","val","test"]:
# for ds in ["test"]:
    # open the proper connection
    writer = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}-{}.tfrecord".format(dataset, ds)))
    writer_small = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}-{}_small.tfrecord".format(dataset, ds)))
    for qn in tqdm(qnames):
        # read the data (numpy)
        dict_data = {}
        for var in vars:
            dict_data[var] = np.load(os.path.join(path_data_in_dataset,"{}_{}_{}.npy".format(ds,qn,var)))
        #wirte it to tensorboard
        for idx in range(dict_data[vars[0]].shape[0]):
            #write the whole lines for a specific dataset
            d_feature = {}
            for var in vars:
                x = dict_data[var][idx]
                d_feature[var] = _floats_feature(x)
            d_feature["deco_enco"] = _floats_feature(id_q[qn])
            features = tf.train.Features(feature=d_feature)
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)
            if idx < 100:
                writer_small.write(serialized)


# For n-2 data

# In[32]:

datasets_ = ["neighbours","random"]
# datasets_ = ["random"]
# datasets_ = ["neighbours"]
# datasets_ = ["two_changes"]
for dataset in datasets_:
    path_data_in_dataset = os.path.join(path_data_in, dataset)
    
    qnames = set([quadnamefromfilename(el) for el in os.listdir(path_data_in_dataset)
                  if os.path.isfile(os.path.join(path_data_in_dataset, el))])
    qnames = np.sort(list(qnames))
    qnames = [q for q in qnames if (q != "computation_infos.json" and q != 'computation_infos_tmp.json')]
    for ds in  ["train","val","test"]:
        # open the proper connection
        writer = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}-{}.tfrecord".format(dataset, ds)))
        writer_small = tf.python_io.TFRecordWriter(os.path.join(path_data_out,"{}-{}_small.tfrecord".format(dataset, ds)))
        for qn in tqdm(qnames):
            # read the data (numpy)
            dict_data = {}
            for var in vars:
                dict_data[var] = np.load(os.path.join(path_data_in_dataset,"{}_{}_{}.npy".format(ds,qn,var)))
            #wirte it to tensorboard
            for idx in range(dict_data[vars[0]].shape[0]):
                #write the whole set for a specific dataset
                d_feature = {}
                for var in vars:
                    x = dict_data[var][idx]
                    d_feature[var] = _floats_feature(x)
                
                qn1, qn2 = qn.split("@")
                tmp = copy.deepcopy(id_q[qn1])
                for id_, el in enumerate(id_q[qn2]):
                    if el:
                        tmp[id_] = el
                d_feature["deco_enco"] = _floats_feature(tmp)
                
                features = tf.train.Features(feature=d_feature)
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)
                if idx < 100:
                    writer_small.write(serialized)


# Check for reading data

# In[46]:

tf.reset_default_graph()
filenames = [os.path.join(path_data_out,"{}.tfrecord".format(ds))]
var = "deco_enco"
print(var)

def _parse_function(example_proto, var, size):
    features = {var: tf.FixedLenFeature((size,), tf.float32, default_value=[0.0 for _ in range(size)]) }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features[var]

dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(lambda x : _parse_function(x,var,dict_data[var].shape[1]))  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


# In[31]:

sess = tf.InteractiveSession()
sess.run(iterator.initializer)

# Start populating the filename queue.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

for i in range(20):
    # Retrieve a single instance:
    x_ = sess.run(next_element)
    print("x_ : {}".format(x_))
#     print("{} \n{}\n\n_______________________".format(x_,dataset_npy[i]))
sess.close()


# In[ ]:




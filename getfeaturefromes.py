from datetime import datetime
from elasticsearch import Elasticsearch
import base64,struct
import numpy as np
import pyflann
import numpy as np
from time import time
from profilehooks import profile
from multiprocessing import Pool
from functools import partial
import json
from clustering import cluster



es = Elasticsearch('10.45.157.120')

feature_cnt=10000
indexname='history_fss_data_v1_1_ycys_data'
querystr={"query":{"bool":{"must":[{
        "range":{"enter_time":{"gt":"2018-06-01T00:00:00.000Z","lt":"2018-06-01T06:00:00.000Z"}}}],
"must_not":[],"should":[]}},
"from":0,"size":feature_cnt,"sort":[],"aggs":{}}

res = es.search(index=indexname, body=querystr)
print("Got %d Hits:" % res['hits']['total'])
#print(res['hits'])
cnt=0
feature=np.zeros((feature_cnt,512))
for hit in res['hits']['hits']:
    rec=hit["_source"]
    #print("%(uuid)s %(rt_feature)s: %(camera_name)s" % rec)
    binfeature=base64.b64decode( rec['rt_feature'])

    #print(binfeature)
    for i in range(int(len(binfeature)/4)-3):
        #print(struct.unpack('f',binfeature[(i+3)*4:(i+4)*4]))
        
        feature[cnt][i]=struct.unpack('f',binfeature[(i+3)*4:(i+4)*4])[0]
    cnt+=1
print(feature.shape)

if __name__ == '__main__':
    descriptor_matrix = feature
    #app_nearest_neighbors, dists = build_index(descriptor_matrix, n_neighbors=2)
    #distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)
    clusters = cluster(descriptor_matrix, n_neighbors=12,thresh=[1,2,3,4,5])
    # print clusters[0]
    clusters_to_be_saved = {}
    for i, cluster in enumerate(clusters[0]["clusters"]):
        c = [int(x) for x in list(cluster)]
        clusters_to_be_saved[i] = c
    with open("clusters.json", "w") as f:
        json.dump(clusters_to_be_saved, f)
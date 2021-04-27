# Advanced-Models-for-Personalized-Product-Search
This repository provides recently advanced personalized models, including: HEM, ZAM(AEM), HRNN.
Other related models will be implimented soon.

### Requirement ###
    waiting to update

### Run: ###
    example: python main.py --structure_name=HEM --score_function=bias_product --device=cuda:0 --emb_dim=200 <br/>
    parameters: </br>
     --structure_name: name of PPS model structure, including "HEM", "ZAM" and "HRNN_simple"
     --score_function: name of function which can measure <u,q,i> score, including "product", "bias_product"
     --data_root: name of datasets root file
     --save_root: name of model.pkl root file
     --device: running device
     --emb_dim: embedding dim of word, item and user
     --att_hidden_units_num: for ZAM, control hidden units of attention matrixs
     --RNN_layers_num: for HRNN, control number of RNN hidden layers
     --batch_size: control batch size
     --EPOCH: control times of training
     --LR: learning rate
     --L2_weight: L2 regularization
     --LAMBDA: personalized weight, [0,1], the weight smaller, the personalization stronger
     --neg_sample_num: number of negative samples
     --noise_rate: control distribution of samples
     --is_val: whether test model on validation set when training
     
### Reference: ###
 ·  Ai Q, Zhang Y, Bi K, et al. Learning a hierarchical embedding model for personalized product search[C]//Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2017: 645-654.
 ·  Ai Q, Hill D N, Vishwanathan S V N, et al. A zero attention model for personalized product search[C]//Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019: 379-388.
 ·  Ge S, Dou Z, Jiang Z, et al. Personalizing search results using hierarchical RNN with query-aware attention[C]//Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2018: 347-356.

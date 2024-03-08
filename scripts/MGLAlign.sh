cd ..

PD=graph_data
TYPE=4
TRAINRATIO=0.1
prefixes=("douban" "allmv_tmdb" "acm_dblp" "phone_email" "foursquare_twitter" "fb-tt" "econ_0.5")
sources=("online" "allmv" "acm" "phone" "foursquare" "facebook" "econ1")
targets=("offline" "tmdb" "dblp" "email" "twitter" "twitter" "econ2")
PREFIX=${prefixes[$TYPE]}
SOURCE=${sources[$TYPE]}
TARGET=${targets[$TYPE]}

TRAIN=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict
TEST=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict
GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth

python network_alignment.py \
    --source_dataset ${PD}/${PREFIX}/${SOURCE}/graphsage/ \
    --target_dataset ${PD}/${PREFIX}/${TARGET}/graphsage/ \
    --groundtruth ${GROUNDTRUTH} \
    --seed 145 \
    --device 7 \
    MGLAlign \
    --cuda \
    --epochs 10 \
    --GCN_output_dim 128 \
    --embedding_decode_dim 128 \
    --feature_embedding_dim 64 \
    --dense_embedding_dim 64 \
    --convergence 500 \
    --local_lr 0.01 \
    --alpha 0.2 \
    --beta 0.8 \
    --gamma 0.5 \
    --_lambda 0.9 \
    --lr 0.005 \
    --ssl_temp 0.8 \
    --top_rate 0.1 \
    --link_topk 5 \
    --num_GCN_blocks 2 \
    --MLP_output_dim 128 \
    --MLP_hidden_dim 128 \
    --weight0 0.25 \
    --weight1 0.25 \
    --weight2 0.25 \
    --weight3 0.25 \
    --weight4 0.5 \
    --weight5 0.5 \
    --weight6 0.5 \
    --embedding_type 'new' \
    --batch_size 256 \
    --act 'tanh' \
    --train_dict ${TRAIN} \
    --test_dict ${TEST} \
    --num_walks 2 \
    --neg_sample_size 20 \
    --walk_len 2 \
    --train_type 'unsuper' \
    --mapping_model 'no' \
    --GNN_hidden_dim 256 \
    --refine_type 'refina' \
    --n_iter 10
: '
--weight0 0.5 \
    --weight1 0.4 \
    --weight2 0.3 \
    --weight3 0.8 \
    --weight4 0.5 \
    --weight5 0.5 \
    --weight6 0.5 \
>> output/MGLAlign/${PREFIX}
for a in 0 0.3 0.5 0.8 1
do
    for b in 0 0.3 0.5 0.8 1
    do
        for c in 0.3 0.5 0.8 1
        do
            PD=graph_data
            TYPE=0
            TRAINRATIO=0.1
            prefixes=("douban" "allmv_tmdb" "weibo_douban")
            sources=("online" "allmv" "weibo")
            targets=("offline" "tmdb" "douban")
            PREFIX=${prefixes[$TYPE]}
            SOURCE=${sources[$TYPE]}
            TARGET=${targets[$TYPE]}

            TRAIN=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict
            TEST=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict
            GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth

            python network_alignment.py \
            --source_dataset ${PD}/${PREFIX}/${SOURCE}/graphsage/ \
            --target_dataset ${PD}/${PREFIX}/${TARGET}/graphsage/ \
            --groundtruth ${GROUNDTRUTH} \
            --seed 145 \
            --device 3 \
            MGLAlign \
            --cuda \
            --epochs 5 \
            --GCN_output_dim 128 \
            --embedding_decode_dim 128 \
            --feature_embedding_dim 32 \
            --convergence 500 \
            --local_lr 0.01 \
            --beta 0.8 \
            --_lambda 0.9 \
            --lr 0.001 \
            --ssl_temp 0.8 \
            --top_rate 0.1 \
            --link_topk 20 \
            --num_GCN_blocks 2 \
            --MLP_output_dim 128 \
            --MLP_hidden_dim 256 \
            --alpha0 ${a} \
            --alpha1 ${b} \
            --alpha2 ${c} \
            --alpha3 0.5 \
            --alpha4 0.5 \
            --alpha5 0.5 \
            --alpha6 0.5 \
            --train_type 'unsuper' \
            --num_walks 5 \
            --walk_len 5  \
            --neg_sample_size 1000 >> output/MGLAlign/${PREFIX}
        done
    done
done
'

: '
#for ((i=1;i<2;i++))
两层GCN,beta=1
有监督，只用最后一层的sup损失，用两层也没什么变化
--epochs 3 \ 3次，5次和2次都比较接近，10次过拟合

有+无，不用增强的无监督，只用最后一层
--epochs 50 \ 比35,70好
Accuracy: 0.3569
MAP: 0.4768
AUC: 0.9882
Precision_1: 0.3551
Precision_5: 0.6172
Precision_10: 0.7308

有+无，不用增强的无监督，用后两层
--epochs 50
Acc: Acc layer 0 is: 0.0733, Acc layer 1 is: 0.2612, Acc layer 2 is: 0.3193, Final acc is: 0.3649
Accuracy: 0.3649
MAP: 0.4806
AUC: 0.9873
Precision_1: 0.3703
Precision_5: 0.6154
Precision_10: 0.7227
Running time: 60.71933341026306

有+无，用增强的无监督，用后两层,beta=1
--epochs 50 比70和35好
Acc: Acc layer 0 is: 0.0733, Acc layer 1 is: 0.2415, Acc layer 2 is: 0.3193, Final acc is: 0.3649
----------------------------------------------------------------------------------------------------
Accuracy: 0.3649
MAP: 0.4792
AUC: 0.9878
Precision_1: 0.3631
Precision_5: 0.6118
Precision_10: 0.7317
----------------------------------------------------------------------------------------------------
Running time: 62.850632190704346
beta=0.8
Acc: Acc layer 0 is: 0.0733, Acc layer 1 is: 0.2370, Acc layer 2 is: 0.3247, Final acc is: 0.3578
----------------------------------------------------------------------------------------------------
Accuracy: 0.3578
MAP: 0.4768
AUC: 0.9879
Precision_1: 0.3569
Precision_5: 0.6127
Precision_10: 0.7335
----------------------------------------------------------------------------------------------------
Running time: 63.86347794532776
beta=0.5,support_loss有助于提升最后一层，但堆起来就更差了
Acc: Acc layer 0 is: 0.0733, Acc layer 1 is: 0.2388, Acc layer 2 is: 0.3256, Final acc is: 0.3506
----------------------------------------------------------------------------------------------------
Accuracy: 0.3506
MAP: 0.4755
AUC: 0.9876
Precision_1: 0.3551
Precision_5: 0.6127
Precision_10: 0.7254
----------------------------------------------------------------------------------------------------
Running time: 64.36122703552246

两层GCN,有+无，用增强的无监督，用后两层,beta=0.8,reg_loss用的是对齐矩阵不是单图嵌入去算对比损失,有一点点提升,epochs=50比80好
Acc: Acc layer 0 is: 0.0733, Acc layer 1 is: 0.2424, Acc layer 2 is: 0.3283, Final acc is: 0.3560
----------------------------------------------------------------------------------------------------
Accuracy: 0.3560
MAP: 0.4798
AUC: 0.9882
Precision_1: 0.3614
Precision_5: 0.6199
Precision_10: 0.7343
----------------------------------------------------------------------------------------------------
Running time: 69.70225310325623

--seed 145 还可以，不是最好
三层GCN用后两层算损失
--alpha0 1 \
--alpha1 0.8 \
--alpha2 1 \
--alpha3 0.5
Accuracy: 0.3596
MAP: 0.4761
AUC: 0.9882
Precision_1: 0.3596
Precision_5: 0.6055
Precision_10: 0.7326

三层GCN用后三层算损失
Accuracy: 0.3623
MAP: 0.4806
AUC: 0.9878
Precision_1: 0.3605
Precision_5: 0.6082
Precision_10: 0.7308

四层GCN用后三层算损失
Accuracy: 0.3685
MAP: 0.4883
AUC: 0.9885
Precision_1: 0.3658
Precision_5: 0.6243
Precision_10: 0.7433

四层GCN用后四层算损失
Accuracy: 0.3721
MAP: 0.4882
AUC: 0.9885
Precision_1: 0.3721
Precision_5: 0.6145
Precision_10: 0.7317

五层GCN用后四层
Accuracy: 0.3792
MAP: 0.4941
AUC: 0.9889
Precision_1: 0.3748
Precision_5: 0.6225
Precision_10: 0.7415

五层GCN用后五层
Accuracy: 0.3909
MAP: 0.5043
AUC: 0.9895
Precision_1: 0.3864
Precision_5: 0.6342
Precision_10: 0.7531

六层GCN用后五层
Accuracy: 0.3864
MAP: 0.4965
AUC: 0.9897
Precision_1: 0.3882
Precision_5: 0.6154
Precision_10: 0.7442

六层GCN用后六层
Accuracy: 0.3801
MAP: 0.4954
AUC: 0.9893
Precision_1: 0.3792
Precision_5: 0.6163
Precision_10: 0.7451

二层GCN
用后两层算损失
--alpha0 1 \ 这个参数1和2差别几乎没有，用1就行
--alpha1 0.8 \ 0.8和0.5各有千秋
--alpha2 1 \ 还是1比较好
Accuracy: 0.3515
MAP: 0.4677
AUC: 0.9868
Precision_1: 0.3533
Precision_5: 0.5921
Precision_10: 0.7182

把--feature_embedding_dim 64->32 
Accuracy: 0.3488
MAP: 0.4703
AUC: 0.9870
Precision_1: 0.3515
Precision_5: 0.5930
Precision_10: 0.7227

把--feature_embedding_dim 64->32 ，再把--top_rate 0.1->0.2 --top_rate 0.2 \--link_topk 10 \
Accuracy: 0.3560
MAP: 0.4730
AUC: 0.9864
Precision_1: 0.3596
Precision_5: 0.5930
Precision_10: 0.7138

把--feature_embedding_dim 64->32 再把--link_topk 10->20 --top_rate 0.1 \--link_topk 20 \

Accuracy: 0.3587
MAP: 0.4708
AUC: 0.9870
Precision_1: 0.3587
Precision_5: 0.5912
Precision_10: 0.7245

--alpha0 1 \ 这个参数1和2差别几乎没有，用1就行
--alpha1 0.5 \
--alpha2 1
Accuracy: 0.3587
MAP: 0.4695
AUC: 0.9868
Precision_1: 0.3614
Precision_5: 0.5886
Precision_10: 0.7156

--cuda \
--epochs 100 \
--GCN_output_dim 128 \
--embedding_decode_dim 128 \
--feature_embedding_dim 64 \
--convergence 500 \
--local_lr 0.01 \
--beta 0.8 \
--_lambda 0.9 \
--lr 0.001 \
--ssl_temp 0.8 \
--top_rate 0.1 \
--link_topk 10 \
--num_GCN_blocks 2 \
--MLP_output_dim 128 \
--MLP_hidden_dim 256

'
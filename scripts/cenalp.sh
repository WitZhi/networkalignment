cd ..

for TYPE in {0..1};
do
    PD=graph_data
    TRAINRATIO=0.1
    prefixes=("douban" "allmv_tmdb")
    sources=("online" "allmv")
    targets=("offline" "tmdb")
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
    CENALP \
    --train_dict ${TRAIN} \
    --num_walks 20 \
    --walk_len 5 \
    --batch_size 512 \
    --threshold 0.5 \
    --linkpred_epochs 0 \
    --num_sample 300 \
    --cuda >> output/CENALP/${PREFIX}
done

#linkpred_epochs表示是否进行链接预测的操作，因为该方法是把对齐和链接预测集成了
# tuning with following notices:
# 1: First set the linkpred_epochs to 0, then tunning other hyper params to achieve the best result.
# 2: num_pair_toadd: Set int, higher value to faster training, but lower accuracy.
# 3: threshold, this is matter only if you use linkpred_epochs > 0, you should set it higher or equal 0.5
# 4: num_sample, this is matter only if you use linkpred_epochs > 0, you sould try 300, 400, ... 1000
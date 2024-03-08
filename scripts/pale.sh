cd ..

for SEED in 111 123 134 222 333
do
    PD=graph_data
    prefixes=("douban" "allmv_tmdb")
    sources=("online" "allmv")
    targets=("offline" "tmdb")
    PREFIX=${prefixes[$TYPE]}
    SOURCE=${sources[$TYPE]}
    TARGET=${targets[$TYPE]}
    TRAINRATIO=0.1
    TRAIN=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict
    GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth

    python network_alignment.py \
    --source_dataset ${PD}/${PREFIX}/${SOURCE}/graphsage/ \
    --target_dataset ${PD}/${PREFIX}/${TARGET}/graphsage/ \
    --groundtruth ${GROUNDTRUTH} \
    --seed ${SEED} \
    PALE \
    --train_dict ${TRAIN} \
    --embedding_dim 300 \
    --embedding_epochs 100 \
    --mapping_epochs 100 \
    --batch_size_embedding 512 \
    --cuda \
    --mapping_model 'linear' >> output/PALE/${PREFIX}
done
#>> output/PALE/${PREFIX}
#GAlign里PALE和CENALP用了10%训练集
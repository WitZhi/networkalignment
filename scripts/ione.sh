
cd ..

for TYPE in 0 1
do
    PD=graph_data
    TRAINRATIO=0.2
    prefixes=("douban" "allmv_tmdb")
    sources=("online" "allmv")
    targets=("offline" "tmdb")
    PREFIX=${prefixes[$TYPE]}
    SOURCE=${sources[$TYPE]}
    TARGET=${targets[$TYPE]}

    TRAIN=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict
    GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/${PREFIX}/${SOURCE}/graphsage/ \
    --target_dataset ${PD}/${PREFIX}/${TARGET}/graphsage/ \
    --groundtruth ${GROUNDTRUTH} \
    IONE \
    --train_dict ${TRAIN} \
    --cuda >> output/IONE/${PREFIX}
done

#>> output/IONE/${PREFIX}

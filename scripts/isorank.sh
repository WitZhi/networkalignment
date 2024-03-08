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
    IsoRank \
    --max_iter 30 \
    --alpha 0.82 \
    --train_dict ${TRAIN} 
done
#>> output/IsoRank/${PREFIX}
#--H ${PD}/${PREFIX}/H.mat \
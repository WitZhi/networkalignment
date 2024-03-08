cd ..

for ((i=1;i<2;i++))
do
    PD=graph_data
    TYPE=0
    TRAINRATIO=0.1
    prefixes=("douban" "allmv_tmdb" "econ")
    sources=("online" "allmv" "econ1")
    targets=("offline" "tmdb" "econ2")
    PREFIX=${prefixes[$TYPE]}
    SOURCE=${sources[$TYPE]}
    TARGET=${targets[$TYPE]}

    TRAIN=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX}/dictionaries/node,split=${TRAINRATIO}.test.dict
    GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth
    MIXUPTYPE=mixup5_5_1

    python network_alignment.py \
    --source_dataset ${PD}/${PREFIX}/${SOURCE}/graphsage/ \
    --target_dataset ${PD}/${PREFIX}/${TARGET}/graphsage/ \
    --groundtruth ${GROUNDTRUTH} \
    GAlign \
    --cuda \
    --GAlign_epochs 20 \
    --refinement_epochs 10 \
    --log \
    --beta 0.8 \
    --graph_mixup \
    --coe_consistency 0.8 \
    --source_mixup_dataset ${PD}/${PREFIX}/${SOURCE}/${MIXUPTYPE}/graphsage/ \
    --target_mixup_dataset ${PD}/${PREFIX}/${TARGET}/${MIXUPTYPE}/graphsage/  \
    --alpha0 1 \
    --alpha1 0.8 \
    --alpha2 0.5 \
    --k 500 \
    --degree_threshold 5\
    --mixup_weight 0 \
    
    : '
    >> output/GAlign/${PREFIX}_norefine_mix
    for THRESHOLD in 2 3 5 7 10
    do
        for K in 5 10 15
        do
            for i in 1 2 3
            do
                MIXUPTYPE=mixup${THRESHOLD}_${K}_${i}

                python network_alignment.py \
                --source_dataset ${PD}/${PREFIX}/${SOURCE}/${MIXUPTYPE}/graphsage/ \
                --target_dataset ${PD}/${PREFIX}/${TARGET}/${MIXUPTYPE}/graphsage/ \
                --groundtruth ${GROUNDTRUTH} \
                GAlign \
                --cuda \
                --log >> output/GAlign/${MIXUPTYPE}_${PREFIX}
            done
        done
    done
    '
done


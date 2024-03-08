cd ..

for SEED in 145
do
    PD=graph_data
    TYPE=0
    TRAINRATIO=0.01
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
    --seed ${SEED} \
    FINAL \
    --H ${PD}/${PREFIX}/H.mat \
    #--train_dict ${TRAIN} 
    
done
#grad-align+里面final和PALE是用了5%的训练集，没有用先验
#Galign里final和Isorank是用了先验H，没有用训练集
#有无训练集差距很大
#--H ${PD}/${PREFIX}/H.mat \
#>> output/FINAL/${PREFIX} ${TRAIN}
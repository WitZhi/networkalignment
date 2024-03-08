cd ..

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
    --device 1 \
    MGLAlign \
    --cuda \
    --train_dict ${TRAIN} 
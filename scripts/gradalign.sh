cd ..

PD=graph_data
TYPE=5
TRAINRATIO=0.1
prefixes=("douban" "allmv_tmdb" "acm_dblp" "phone_email" "econ" "fb-tt") 
sources=("online" "allmv" "acm" "phone" "econ1" "facebook")
targets=("offline" "tmdb" "dblp" "email" "econ2" "twitter")
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
    GradAlignPlus \
    --train_dict ${TRAIN} \
    --test_dict ${TEST} \
    --cuda \
    --epochs 10 \
    --train_type 'unsuper' \
    --lr 0.001 \
    --alpha0 0.25 \
    --alpha1 0.25 \
    --alpha2 0.25 \
    --alpha3 0.25 \
    --alpha4 0.5 \
    --alpha5 0.5 \
    --alpha6 0.5 \
    --hid_dim 100 
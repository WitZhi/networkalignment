cd ..

PD=graph_data
prefixes=("douban" "allmv_tmdb")
sources=("online" "allmv")
targets=("offline" "tmdb")
PREFIX=${prefixes[$TYPE]}
SOURCE=${sources[$TYPE]}
TARGET=${targets[$TYPE]}

SOURCE=${PD}/${PREFIX}/${SOURCE}/shuffle/graphsage/
TARGET=${PD}/${PREFIX}/${TARGET}/graphsage/
GROUNDTRUTH=${PD}/${PREFIX}/dictionaries/groundtruth
STATISTICS=${PD}/${PREFIX}/source_shuffle/statistics/

python dataset.py \
    --source_dataset ${SOURCE} \
    --target_dataset ${TARGET} \
    --groundtruth ${GROUNDTRUTH} \
    --output_dir ${STATISTICS} \
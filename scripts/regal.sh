cd ..

#allmv_tmdb的节点索引映射缺失，导致对于谱分析方法不能直接使用
PD=graph_data
TRAINRATIO=0.1
TYPE=0
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
REGAL >> output/REGAL/${PREFIX}


: <<'COMMENT'
> ${directory_name}/${PREFIX}
PD=$HOME/dataspace/graph/flickr_lastfm
PREFIX1=flickr
PREFIX2=lastfm

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
REGAL > output/REGAL/fl




PD=$HOME/dataspace/graph/flickr_myspace
PREFIX1=flickr
PREFIX2=myspace

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
REGAL > output/REGAL/fm



PD=$HOME/dataspace/graph/fb-tw-data
PREFIX1=facebook
PREFIX2=twitter

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
REGAL > output/REGAL/fb_tw



PD=$HOME/dataspace/graph/fq-tw-data
PREFIX1=foursquare
PREFIX2=twitter

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
REGAL > output/REGAL/fq_tw




PD=$HOME/dataspace/graph/arenas
PREFIX1=arenas1
PREFIX2=arenas2

python network_alignment.py \
--source_dataset ${PD}/${PREFIX1}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
REGAL > output/REGAL/arenas




PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
PREFIX1=ppi
PREFIX2=ppi

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/permut/graphsage/ \
--groundtruth ${PD}/permut/dictionaries/groundtruth \
REGAL > output/REGAL/ppi
COMMENT




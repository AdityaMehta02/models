if [ $# -lt 2 ]; then
	echo "Usage <dir> <filename>"
        exit 0
fi

FLIST=`cat $2`

mkdir -p PredictIOU/$1/JPEGImages
mkdir -p PredictIOU/$1/SegmentationClass

for fname in ${FLIST}
do
    F_JPEG=$fname'.jpg'
    cp IOUImages/JPEGImages/${F_JPEG} PredictIOU/$1/JPEGImages/
    F_PNG=$fname'.png'
    cp IOUImages/Segmentation/${F_PNG} PredictIOU/$1/SegmentationClass/
done

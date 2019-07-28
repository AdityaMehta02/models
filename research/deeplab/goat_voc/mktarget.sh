FLIST=`cat $1`
for fname in ${FLIST}
do
    cp IOUImages/JPEGImages/${fname} TargetIOU/JPEGImages/
    SEG=`echo $fname | awk -F '.' '{print $1}'`
    SEG_PNG=$SEG'.png'
    #echo ${SEG_PNG}
    cp IOUImages/Segmentation/${SEG_PNG} TargetIOU/SegmentationClass/
done

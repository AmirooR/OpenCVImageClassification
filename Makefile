OPENCV_CFLAGS=`pkg-config --cflags opencv` -I/opt/local/include/opencv2
OPENCV_LFLAGS=`pkg-config --libs opencv`

all: acc testRead runKmeans kdtreeIndex

acc: drwnHOGFeatures.o AccumulateFeatures.cpp FeatureExtractor.h DataSetReader.h
	g++ -g -O3 -m32 ${OPENCV_CFLAGS} ${OPENCV_LFLAGS}  drwnHOGFeatures.o AccumulateFeatures.cpp -o acc

testRead: testRead.cpp
	g++ -g  ${OPENCV_CFLAGS} ${OPENCV_LFLAGS} testRead.cpp -o testRead

runKmeans: runKmeans.cpp
	g++ -g  ${OPENCV_CFLAGS} ${OPENCV_LFLAGS} runKmeans.cpp -o runKmeans

kdtreeIndex: kdtreeIndex.cpp FeatureExtractor.h DataSetReader.h drwnHOGFeatures.o
	g++ -g  ${OPENCV_CFLAGS} ${OPENCV_LFLAGS} kdtreeIndex.cpp drwnHOGFeatures.o  -o kdtreeIndex

drwnHOGFeatures.o: drwnHOGFeatures.cpp drwnHOGFeatures.h
	g++ -g -c ${OPENCV_CFLAGS} drwnHOGFeatures.cpp -o drwnHOGFeatures.o

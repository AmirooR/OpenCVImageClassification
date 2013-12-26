/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Copyright (c) 2007-2012, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnHOGFeatures.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

//#include "stdafx.h"
#include <cstdlib>
#include <vector>

#include <cv.h>
#include <cstdlib>

#include <cxcore.h>
#include <highgui.h>
//#include <opencv.hpp>
//#include "drwnBase.h"
#include "drwnHOGFeatures.h"
//#include "drwnOpenCVUtils.h"

// drwnHOGFeatures static members -------------------------------------------

int drwnHOGFeatures::DEFAULT_CELL_SIZE = 8;
int drwnHOGFeatures::DEFAULT_BLOCK_SIZE = 2;
int drwnHOGFeatures::DEFAULT_BLOCK_STEP = 1;
int drwnHOGFeatures::DEFAULT_ORIENTATIONS = 9;
drwnHOGNormalization drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L2_NORM;
double drwnHOGFeatures::DEFAULT_CLIPPING_LB = 0.1;
double drwnHOGFeatures::DEFAULT_CLIPPING_UB = 0.5;
bool drwnHOGFeatures::DEFAULT_DIM_REDUCTION = false;
// convert image to greyscale (floating point)
IplImage *drwnGreyImage(const IplImage *src)
{
	
	assert(src != NULL);

	IplImage *dst = NULL;
	if (src->nChannels == 3) {
		IplImage *tmpImg = cvCreateImage(cvGetSize(src), src->depth, 1);
		cvCvtColor(src, tmpImg, CV_RGB2GRAY);
		if (tmpImg->depth == IPL_DEPTH_8U) {
			dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
			cvConvertScale(tmpImg, dst, 1.0 / 255.0);
			cvReleaseImage(&tmpImg);
		} else {
			assert(tmpImg->depth == IPL_DEPTH_32F);
			dst = tmpImg;
		}
	} else {
		if (src->depth == IPL_DEPTH_8U) {
			dst = cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
			cvConvertScale(src, dst, 1.0 / 255.0);
		} else {
			assert(src->depth == IPL_DEPTH_32F);
			dst = cvCloneImage(src);
		}
	}

	return dst;
}

// pad image and copy boundary
IplImage *drwnPadImage(IplImage *src, int margin)
{   
	assert((src != NULL) && (margin > 0));
	return drwnPadImage(src,
			cvRect(-margin/2, -margin/2, src->width + margin, src->height + margin));
}

IplImage *drwnPadImage(const IplImage *src, int margin)
{   
	assert(src != NULL);
	IplImage *imgCpy = cvCloneImage(src);
	IplImage *paddedImg = drwnPadImage(imgCpy, margin);
	cvReleaseImage(&imgCpy);
	return paddedImg;
}
    
IplImage *drwnPadImage(IplImage *src, const CvRect& page)
{   
	assert((src != 0) && (page.x <= 0) && (page.y <= 0) &&
			(page.x + page.width >= src->width) && (page.y + page.height >= src->height));
	IplImage *paddedImg = cvCreateImage(cvSize(page.width, page.height),
			src->depth, src->nChannels);

	// copy image to (0,0) of page
	cvSetImageROI(paddedImg, cvRect(-page.x, -page.y, src->width, src->height));
	cvCopyImage(src, paddedImg);
	cvResetImageROI(paddedImg);
	// pad with mirroring
	if (page.x < 0) {
		cvSetImageROI(src, cvRect(0, 0, -page.x, src->height));
		cvSetImageROI(paddedImg, cvRect(0, -page.y, -page.x, src->height));
		cvFlip(src, paddedImg, 1);
	}

	if (page.x + page.width > src->width) {
		const int M = page.width + page.x - src->width;
		cvSetImageROI(src, cvRect(src->width - M, 0, M, src->height));
		cvSetImageROI(paddedImg, cvRect(page.width - M, -page.y, M, src->height));
		cvFlip(src, paddedImg, 1);
	}

	if (page.y < 0) {
		cvSetImageROI(src, cvRect(0, 0, src->width, -page.y));
		cvSetImageROI(paddedImg, cvRect(-page.x, 0, src->width, -page.y));
		cvFlip(src, paddedImg, 0);
	}

	if (page.y + page.height > src->height) {
		const int M = page.height + page.y - src->height;
		cvSetImageROI(src, cvRect(0, src->height - M, src->width, M));
		cvSetImageROI(paddedImg, cvRect(-page.x, page.height - M, src->width, M));
		cvFlip(src, paddedImg, 0);
	}

	if ((page.x < 0) && (page.y < 0)) {
		cvSetImageROI(src, cvRect(0, 0, -page.x, -page.y));
		cvSetImageROI(paddedImg, cvRect(0, 0, -page.x, -page.y));
		cvFlip(src, paddedImg, -1);
	}
	if ((page.x < 0) && (page.y + page.height > src->height)) {
		const int M = page.height + page.y - src->height;
		cvSetImageROI(src, cvRect(0, src->height - M, -page.x, M));
		cvSetImageROI(paddedImg, cvRect(0, page.height - M, -page.x, M));
		cvFlip(src, paddedImg, -1);
	}

	if ((page.x + page.width > src->width) && (page.y < 0)) {
		const int M = page.width + page.x - src->width;
		cvSetImageROI(src, cvRect(src->width - M, 0, M, -page.y));
		cvSetImageROI(paddedImg, cvRect(page.width - M, 0, M, -page.y));
		cvFlip(src, paddedImg, -1);
	}

	if ((page.x + page.width > src->width) && (page.y + page.height > src->height)) {
		const int M = page.width + page.x - src->width;
		const int N = page.height + page.y - src->height;
		cvSetImageROI(src, cvRect(src->width - M, src->height - N, M, N));
		cvSetImageROI(paddedImg, cvRect(page.width - M, page.height - N, M, N));
		cvFlip(src, paddedImg, -1);
	}

	// reset ROIs
	cvResetImageROI(src);
	cvResetImageROI(paddedImg);

	return paddedImg;
}

IplImage *drwnPadImage(const IplImage *src, const CvRect& page)
{
	assert(src != NULL);
	IplImage *imgCpy = cvCloneImage(src);
	IplImage *paddedImg = drwnPadImage(imgCpy, page);
	cvReleaseImage(&imgCpy);
	return paddedImg;
}

void releaseOpenCVMatrices(vector<CvMat *>& matrices)
{
	for (int i = 0; i < (int)matrices.size(); i++) {
		if (matrices[i] != NULL) {
			cvReleaseMat(&matrices[i]);
			matrices[i] = NULL;
		}
	}
}

void releaseOpenCVMatrices(pair<CvMat *, CvMat *>& matrices)
{
	if (matrices.second != NULL)
		cvReleaseMat(&matrices.second);
	if (matrices.first != NULL)
		cvReleaseMat(&matrices.first);
}

pair<CvMat *, CvMat *> createOpenCVPair(int rows, int cols, int mTypeFirst,
		    int mTypeSecond)
{
	std::pair<CvMat *, CvMat *> matrices;
	matrices.first = cvCreateMat(rows, cols, mTypeFirst);
	assert(matrices.first != NULL);
	matrices.second = cvCreateMat(rows, cols, mTypeSecond);
	assert(matrices.second != NULL);

	return matrices;
}

pair<CvMat *, CvMat *> createOpenCVPair(int rows, int cols, int mType)
{
	return createOpenCVPair(rows, cols, mType, mType);
}



// drwnHOGFeatures ----------------------------------------------------------

drwnHOGFeatures::drwnHOGFeatures() :
    _cellSize(DEFAULT_CELL_SIZE), _blockSize(DEFAULT_BLOCK_SIZE),
    _blockStep(DEFAULT_BLOCK_STEP), _numOrientations(DEFAULT_ORIENTATIONS),
    _normalization(DEFAULT_NORMALIZATION),
    _clipping(DEFAULT_CLIPPING_LB, DEFAULT_CLIPPING_UB),
    _bDimReduction(DEFAULT_DIM_REDUCTION)
{
    // do nothing
}

drwnHOGFeatures::~drwnHOGFeatures()
{
    // do nothing
}

// gradient pre-processing (can be provided to computeFeatures)
pair<CvMat *, CvMat *> drwnHOGFeatures::gradientMagnitudeAndOrientation(const IplImage *img) const
{
    assert((img != NULL) && (img->nChannels == 1));

    //DRWN_LOG_DEBUG("Computing gradients on " << toString(*img) << "...");
    pair<CvMat *, CvMat *> gradients = createOpenCVPair(img->height, img->width, CV_32FC1);
    cvSobel(img, gradients.first, 1, 0, 3);
    cvSobel(img, gradients.second, 0, 1, 3);

    // get canonical orientations for quantizing
    vector<float> u, v;
    computeCanonicalOrientations(u, v);

    // allocate memory for magintude (float) and orientation (int)
    CvMat *gradMagnitude = cvCreateMat(img->height, img->width, CV_32FC1);
    CvMat *gradOrientation = cvCreateMat(img->height, img->width, CV_32SC1);

    const float *pDx = (const float *)CV_MAT_ELEM_PTR(*gradients.first, 0, 0);
    const float *pDy = (const float *)CV_MAT_ELEM_PTR(*gradients.second, 0, 0);
    float *pm = (float *)CV_MAT_ELEM_PTR(*gradMagnitude, 0, 0);
    int *po = (int *)CV_MAT_ELEM_PTR(*gradOrientation, 0, 0);
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++, pDx++, pDy++, pm++, po++) {
            // madnitude
            *pm = sqrt(((*pDx) * (*pDx)) + ((*pDy) * (*pDy)));

            // orientation
            *po = 0;
            float bestScore = fabs(u[0] * (*pDx) + v[0] * (*pDy));
            for (int i = 1; i < _numOrientations; i++) {
                float score = fabs(u[i] * (*pDx) + v[i] * (*pDy));
                if (score > bestScore) {
                    bestScore = score;
                    *po = i;
                }
            }
        }
    }

    releaseOpenCVMatrices(gradients);
    return make_pair(gradMagnitude, gradOrientation);
}

// feature computation
void drwnHOGFeatures::computeFeatures(const IplImage *img, vector<CvMat *> &features)
{
    // check input
    assert(img != NULL);

    // color convert
    //DRWN_LOG_DEBUG("Color converting " << img->width << "-by-" << img->height << " image...");
    IplImage *greyImg = drwnGreyImage(img);

    // compute and quantize gradients
    pair<CvMat *, CvMat *> magAndOri = gradientMagnitudeAndOrientation(img);

    // compute actual features
    computeFeatures(magAndOri, features);

    // free memory
    releaseOpenCVMatrices(magAndOri);
    cvReleaseImage(&greyImg);
}

void drwnHOGFeatures::computeFeatures(const pair<CvMat *, CvMat *>& gradMagAndOri,
    vector<CvMat *>& features)
{
    //DRWN_FCN_TIC;
    const int NUM_FEATURES = numFeatures();

    // check input
    assert((gradMagAndOri.first != NULL) && (gradMagAndOri.second != NULL));
    assert((gradMagAndOri.first->rows == gradMagAndOri.second->rows) &&
        (gradMagAndOri.first->cols == gradMagAndOri.second->cols));
    assert(cvGetElemType(gradMagAndOri.first) == CV_32FC1);
    assert(cvGetElemType(gradMagAndOri.second) == CV_32SC1);

    if (features.empty()) {
        features.resize(NUM_FEATURES, (CvMat *)NULL);
    }
    assert((int)features.size() == NUM_FEATURES);

    // compute cell histograms
    vector<CvMat *> cellHistograms(_numOrientations, (CvMat *)NULL);
    computeCellHistograms(gradMagAndOri, cellHistograms);

    // group into blocks and normalize
    vector<CvMat *> *featuresPtr = NULL;
    vector<CvMat *> fullFeatures;
    if (_bDimReduction) {
        fullFeatures.resize(_blockSize * _blockSize * _numOrientations, (CvMat *)NULL);
        featuresPtr = &fullFeatures;
    } else {
        featuresPtr = &features;
    }
    computeBlockFeatures(cellHistograms, *featuresPtr);

    // normalize
    normalizeFeatureVectors(*featuresPtr);
    if ((_clipping.first > 0.0) || (_clipping.second < 1.0)) {
        clipFeatureVectors(*featuresPtr);
        normalizeFeatureVectors(*featuresPtr);
    }

    // reduce dimensionality
    if (_bDimReduction) {
        for (int i = 0; i < numFeatures(); i++) {
            if ((features[i] != NULL) && ((features[i]->rows != fullFeatures[0]->rows) ||
                    (features[i]->cols != fullFeatures[0]->cols))) {
                cvReleaseMat(&features[i]);
            }
            if (features[i] == NULL) {
                features[i] = cvCreateMat(fullFeatures[0]->rows, fullFeatures[0]->cols, CV_32FC1);
            }
            cvZero(features[i]);
        }

        // energy and orientation sums
        for (int i = 0; i < _numOrientations; i++) {
            for (int j = 0; j < _blockSize * _blockSize; j++) {
                cvAdd(features[i], fullFeatures[j * _numOrientations + i], features[i]);
                cvAdd(features[_numOrientations + j], fullFeatures[j * _numOrientations + i],
                    features[_numOrientations + j]);
            }
        }
    }

    // free memory
    releaseOpenCVMatrices(fullFeatures);
    releaseOpenCVMatrices(cellHistograms);

    //DRWN_FCN_TOC;
}

// dense features are associated with the pixel at the center of the block
void drwnHOGFeatures::computeDenseFeatures(const IplImage *img, vector<IplImage *> &features)
{
    //DRWN_FCN_TIC;
    assert(features.empty());

#if 0
    vector<CvMat *> responses;
    computeFeatures(img, responses);

    for (unsigned i = 0; i < responses.size(); i++) {
        IplImage *m = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
        cvResize(responses[i], m);
        cvReleaseMat(&responses[i]);
        features.push_back(m);
    }
#else

    // initialize features
    for (int i = 0; i < numFeatures(); i++) {
        features.push_back(cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1));
        cvZero(features.back());
    }

    // first pad image by blockSize/2 border
    IplImage *greyImg = drwnGreyImage(img);
    IplImage *paddedImg = drwnPadImage(greyImg, _blockSize * _cellSize);
    //drwnShowDebuggingImage(paddedImg, "paddedImage", true);

    // force single cell stepping
    int oldBlockStep = _blockStep;
    _blockStep = 1;

    // compute gradients
    pair<CvMat *, CvMat *> magAndOri = gradientMagnitudeAndOrientation(paddedImg);

    // compute features with shifted origins
    vector<CvMat *> responses(numFeatures(), (CvMat *)NULL);
    CvMat *shiftedMag = cvCreateMat(greyImg->height, greyImg->width, CV_32FC1);
    CvMat *shiftedOri = cvCreateMat(greyImg->height, greyImg->width, CV_32SC1);
    for (int y = 0; y < _cellSize; y++) {
        for (int x = 0; x < _cellSize; x++) {

            //DRWN_LOG_DEBUG("...copying from " << toString(*magAndOri.first) << " to "
             //   << toString(*shiftedMag) << " at offset (" << x << ", " << y << ")");

            // shift gradients
            CvMat subBlock;
            cvGetSubRect(magAndOri.first, &subBlock, cvRect(x, y, greyImg->width, greyImg->height));
            cvCopy(&subBlock, shiftedMag);
            cvGetSubRect(magAndOri.second, &subBlock, cvRect(x, y, greyImg->width, greyImg->height));
            cvCopy(&subBlock, shiftedOri);

            // compute features
            computeFeatures(make_pair(shiftedMag, shiftedOri), responses);

            // copy into feature locations
            for (unsigned i = 0; i < responses.size(); i++) {
                for (int yy = 0; yy < responses[i]->height; yy++) {
                    if (yy * _cellSize + y >= greyImg->height) break;
                    float *p = &CV_IMAGE_ELEM(features[i], float, yy * _cellSize + y, x);
                    const float *q = (const float *)CV_MAT_ELEM_PTR(*responses[i], yy, 0);
                    for (int xx = 0; xx < responses[i]->width; xx++) {
                        if (xx * _cellSize + x >= greyImg->width) break;
                        p[xx * _cellSize] = q[xx];
                    }
                }
            }
        }
    }

    // restore state
    _blockStep = oldBlockStep;

    // free memory
    releaseOpenCVMatrices(responses);
    cvReleaseMat(&shiftedOri);
    cvReleaseMat(&shiftedMag);
    releaseOpenCVMatrices(magAndOri);
    cvReleaseImage(&paddedImg);
    cvReleaseImage(&greyImg);
#endif

    //DRWN_FCN_TOC;
}

// visualization
IplImage *drwnHOGFeatures::visualizeCells(const IplImage *img, int scale)
{
    if (scale < 1) scale = 1;

    // TODO: refactor to share computation
    int oldSize = _blockSize;
    int oldStep = _blockStep;

    _blockSize = 1;
    _blockStep = 1;

    vector<CvMat *> features;
    computeFeatures(img, features);
    assert(features.size() == (unsigned)_numOrientations);

    vector<float> u, v;
    computeCanonicalOrientations(u, v);

    int numCellsX = features[0]->cols;
    int numCellsY = features[0]->rows;

    IplImage *canvas = cvCreateImage(cvSize(scale * numCellsX * _cellSize, scale * numCellsY * _cellSize),
        IPL_DEPTH_8U, 3);
    cvZero(canvas);

    for (int y = 0; y < numCellsY; y++) {
        int cy = scale * (y * _cellSize + _cellSize/2);
        for (int x = 0; x < numCellsX; x++) {
            int cx = scale * (x * _cellSize + _cellSize/2);

            multimap<float, int> sortedOrientations;
            for (int o = 0; o < _numOrientations; o++) {
                sortedOrientations.insert(make_pair(CV_MAT_ELEM(*features[o], float, y, x), o));
            }

            // draw orientations in sorted order
            for (multimap<float, int>::const_iterator it = sortedOrientations.begin();
                 it != sortedOrientations.end(); it++) {
                int strength = (int)(255.0 * it->first);
                int o = it->second;

                // rotate gradient orientations by 90 degrees
                cvLine(canvas, cvPoint((int)(cx - 0.9 * scale * v[o] * _cellSize/2),
                        (int)(cy + 0.9 * scale * u[o] * _cellSize/2)),
                    cvPoint((int)(cx + 0.9 * scale * v[o] * _cellSize/2),
                        (int)(cy - 0.9 * scale * u[o] * _cellSize/2)),
                    CV_RGB(strength, strength, strength), std::min(2, scale));
            }
        }
    }

    releaseOpenCVMatrices(features);
    _blockSize = oldSize;
    _blockStep = oldStep;
    return canvas;
}

// compute the representative orientation for each bin
void drwnHOGFeatures::computeCanonicalOrientations(vector<float>& x, vector<float>& y) const
{
    x.resize(_numOrientations);
    y.resize(_numOrientations);

    double theta = 0.0;
    for (int i = 0; i < _numOrientations; i++) {
        x[i] = cos(theta);
        y[i] = sin(theta);
        theta += M_PI / (_numOrientations + 1.0);
    }
    assert(theta <= M_PI);
}

// compute histograms of oriented gradients for each cell
void drwnHOGFeatures::computeCellHistograms(const pair<CvMat *, CvMat *>& gradMagAndOri,
    vector<CvMat *>& cellHistograms) const
{
    // check input
    const CvMat *gradMagnitude = gradMagAndOri.first;
    const CvMat *gradOrientation = gradMagAndOri.second;
    assert((gradMagnitude != NULL) && (gradOrientation != NULL));
    assert((gradMagnitude->rows == gradOrientation->rows) &&
        (gradMagnitude->cols == gradOrientation->cols));

    const int width = gradMagnitude->cols;
    const int height = gradMagnitude->rows;

    // compute cell histograms
    const int numCellsX = (width + _cellSize - 1) / _cellSize;
    const int numCellsY = (height + _cellSize - 1) / _cellSize;
    //DRWN_LOG_DEBUG("Computing histograms for each " << numCellsX << "-by-" << numCellsY << " cell...");

    // TODO: use vector create helper method
    assert(cellHistograms.size() == (unsigned)_numOrientations);
    for (int i = 0; i < _numOrientations; i++) {
        if (cellHistograms[i] == NULL) {
            cellHistograms[i] = cvCreateMat(numCellsY, numCellsX, CV_32FC1);
        }
        cvZero(cellHistograms[i]);
    }

    // vote for cell using interpolation
    for (int y = 0; y < height; y++) {
        const float *pm = (const float *)CV_MAT_ELEM_PTR(*gradMagnitude, y, 0);
        const int *po = (const int *)CV_MAT_ELEM_PTR(*gradOrientation, y, 0);
        for (int x = 0; x < width; x++) {

            float cellXCoord = ((float)x + 0.5) / (float)_cellSize - 0.5;
            float cellYCoord = ((float)y + 0.5) / (float)_cellSize - 0.5;

            int cellXIndx = (int)cellXCoord; // integer cell index
            int cellYIndx = (int)cellYCoord;

            cellXCoord -= (float)cellXIndx; // fractional cell index
            cellYCoord -= (float)cellYIndx;

            float *pc = (float *)CV_MAT_ELEM_PTR(*cellHistograms[po[x]], cellYIndx, cellXIndx);

            if ((cellXIndx >= 0) && (cellYIndx >= 0)) {
                pc[0] += (1.0f - cellXCoord) * (1.0f - cellYCoord) * pm[x];
            }

            if ((cellXIndx >= 0) && (cellYIndx + 1 < numCellsY)) {
                pc[numCellsX] += (1.0f - cellXCoord) * cellYCoord * pm[x];
            }

            if ((cellXIndx + 1 < numCellsX) && (cellYIndx + 1 < numCellsY)) {
                pc[numCellsX + 1] += cellXCoord * cellYCoord * pm[x];
            }

            if ((cellXIndx + 1 < numCellsX) && (cellYIndx >= 0)) {
                pc[1] += cellXCoord * (1.0f - cellYCoord) * pm[x];
            }
        }
    }

#if 0
    // debugging
    for (int i = 0; i < _numOrientations; i++) {
        string name = string("histogram ") + toString(i);
        drwnShowDebuggingImage(cellHistograms[i], name.c_str(), false);
    }
    cvWaitKey(-1);
#endif
}

void drwnHOGFeatures::computeBlockFeatures(const vector<CvMat *>& cellHistograms,
    vector<CvMat *>& features) const
{
    const int NUM_FEATURES = _blockSize * _blockSize * _numOrientations;
    assert(features.size() == (unsigned)NUM_FEATURES);
    assert(cellHistograms.size() == (unsigned)_numOrientations);

    // group cell histograms into blocks
    const int numCellsX = cellHistograms.front()->width;
    const int numCellsY = cellHistograms.front()->height;
    const int numBlocksX = (numCellsX - _blockSize + 1) / _blockStep;
    const int numBlocksY = (numCellsY - _blockSize + 1) / _blockStep;

    //DRWN_LOG_DEBUG("Computing and normalizing feature vectors for " <<
     //   numBlocksX << "-by-" << numBlocksY << " blocks...");

    // allocate memory for blocks
    for (int i = 0; i < NUM_FEATURES; i++) {
        if ((features[i] != NULL) && ((features[i]->rows != numBlocksY) ||
                (features[i]->cols != numBlocksX))) {
            cvReleaseMat(&features[i]);
        }
        if (features[i] == NULL) {
            features[i] = cvCreateMat(numBlocksY, numBlocksX, CV_32FC1);
        }
    }

    // compute features
    for (int y = 0; y < numBlocksY; y++) {
        for (int x = 0; x < numBlocksX; x++) {

            int featureIndx = 0;
            for (int cy = _blockStep * y; cy < _blockStep * y + _blockSize; cy++) {
                for (int cx = _blockStep * x; cx < _blockStep * x + _blockSize; cx++) {
                    for (int o = 0; o < _numOrientations; o++) {
                        CV_MAT_ELEM(*features[featureIndx], float, y, x) =
                            CV_MAT_ELEM(*cellHistograms[o], float, cy, cx);
                        featureIndx += 1;
                    }
                }
            }
        }
    }
}

// normalization
void drwnHOGFeatures::normalizeFeatureVectors(vector<CvMat *>& features) const
{
    if (features.empty()) return;

    int height = features[0]->rows;
    int width = features[0]->cols;

    // compute normalization constant
    CvMat *Z = cvCreateMat(height, width, CV_32FC1);
    CvMat *tmp = cvCreateMat(height, width, CV_32FC1);
    cvSet(Z, cvScalar(DRWN_EPSILON));

    for (unsigned i = 0; i < features.size(); i++) {
        switch (_normalization) {
        case DRWN_HOG_L2_NORM:
            cvMul(features[i], features[i], tmp);
            break;
        case DRWN_HOG_L1_NORM:
        case DRWN_HOG_L1_SQRT:
            cvAbs(features[i], tmp);
            break;
        default:
	    std::cerr<<"Fatal: unknown normalization method"<<std::endl;
            //DRWN_LOG_FATAL("unknown normalization method");
        }

        cvAdd(Z, tmp, Z);
    }

    if (_normalization == DRWN_HOG_L2_NORM) {
        cvPow(Z, Z, 0.5);
    }

    // perform normalization
    for (unsigned i = 0; i < features.size(); i++) {
        cvDiv(features[i], Z, features[i]);
    }

    if (_normalization == DRWN_HOG_L1_SQRT) {
        for (unsigned i = 0; i < features.size(); i++) {
            cvPow(features[i], features[i], 0.5);
        }
    }

    // free memory
    cvReleaseMat(&tmp);
    cvReleaseMat(&Z);
}

// clip feature values to [C.lb, C.ub] and rescale to [0, 1]
void drwnHOGFeatures::clipFeatureVectors(std::vector<CvMat *>& features) const
{
    assert(_clipping.first < _clipping.second);
    for (unsigned i = 0; i < features.size(); i++) {
        cvMaxS(features[i], _clipping.first, features[i]);
        cvMinS(features[i], _clipping.second, features[i]);
        cvConvertScale(features[i], features[i], 1.0 / (_clipping.second - _clipping.first),
            - _clipping.first / (_clipping.second - _clipping.first));
    }
}

// drwnHOGFeaturesConfig ----------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnHOGFeatures
//! \b cellSize        :: cell size in pixels (default: 8)\n
//! \b blockSize       :: block size in cells (default: 2)\n
//! \b blockStep       :: block step in cells (default: 1)\n
//! \b numOrientations :: number of orientations (default: 9)\n
//! \b normMethod      :: normalization method (L2_NORM (default), L1_NORM, L1_SQRT)\n
//! \b normClippingLB  :: lower clipping after normalization (default: 0.1)\n
//! \b normClippingUB  :: upper clipping after normalization (default: 0.5)\n
//! \b dimReduction    :: analytic dimensionality reduction (default: false)
/*
class drwnHOGFeaturesConfig : public drwnConfigurableModule {
public:
    drwnHOGFeaturesConfig() : drwnConfigurableModule("drwnHOGFeatures") { }
    ~drwnHOGFeaturesConfig() { }

    void usage(ostream &os) const {
        os << "      cellSize        :: cell size in pixels (default: "
           << drwnHOGFeatures::DEFAULT_CELL_SIZE << ")\n";
        os << "      blockSize       :: block size in cells (default: "
           << drwnHOGFeatures::DEFAULT_BLOCK_SIZE << ")\n";
        os << "      blockStep       :: block step in cells (default: "
           << drwnHOGFeatures::DEFAULT_BLOCK_STEP << ")\n";
        os << "      numOrientations :: number of orientations (default: "
           << drwnHOGFeatures::DEFAULT_ORIENTATIONS << ")\n";
        os << "      normMethod      :: normalization method (L2_NORM (default), L1_NORM, L1_SQRT)\n";
        os << "      normClippingLB  :: lower clipping after normalization (default: "
           << drwnHOGFeatures::DEFAULT_CLIPPING_LB << ")\n";
        os << "      normClippingUB  :: upper clipping after normalization (default: "
           << drwnHOGFeatures::DEFAULT_CLIPPING_UB << ")\n";
        os << "      dimReduction    :: analytic dimensionality reduction (default: "
           << (drwnHOGFeatures::DEFAULT_DIM_REDUCTION ? "true" : "false") << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "cellSize")) {
            drwnHOGFeatures::DEFAULT_CELL_SIZE = std::max(1, atoi(value));
        } else if (!strcmp(name, "blockSize")) {
            drwnHOGFeatures::DEFAULT_BLOCK_SIZE = std::max(1, atoi(value));
        } else if (!strcmp(name, "blockStep")) {
            drwnHOGFeatures::DEFAULT_BLOCK_STEP = std::max(1, atoi(value));
        } else if (!strcmp(name, "numOrientations")) {
            drwnHOGFeatures::DEFAULT_ORIENTATIONS = std::max(1, atoi(value));
        } else if (!strcmp(name, "normMethod")) {
            if (!strcasecmp(value, "L2_NORM")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L2_NORM;
            } else if (!strcasecmp(value, "L1_NORM")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L1_NORM;
            } else if (!strcasecmp(value, "L1_SQRT")) {
                drwnHOGFeatures::DEFAULT_NORMALIZATION = DRWN_HOG_L1_SQRT;
            } else {
                //DRWN_LOG_FATAL("unrecognized configuration value " << value
                 //   << " for option " << name << " in " << this->name());
		    std::cerr<<"Fatal unrecognized configuration value "<<std::endl;
            }
        } else if (!strcmp(name, "normClippingLB")) {
            drwnHOGFeatures::DEFAULT_CLIPPING_LB = std::max(0.0, atof(value));
        } else if (!strcmp(name, "normClippingUB")) {
            drwnHOGFeatures::DEFAULT_CLIPPING_UB = std::min(1.0, atof(value));
        } else if (!strcmp(name, "dimReduction")) {
            drwnHOGFeatures::DEFAULT_DIM_REDUCTION = trueString(string(value));
        } else {
            //DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
		std::cerr<<"Fatal unrecognized configuration option name"<<std::endl;
        }
    }
};

static drwnHOGFeaturesConfig gHOGFeaturesConfig;*/

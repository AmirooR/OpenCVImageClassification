/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Copyright (c) 2007-2012, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnHOGFeatures.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#pragma once

//#include "stdafx.h"
#include <vector>
#include <iostream>
#include "math.h"
//#include "stdlib.h"
//#define M_PI 3.141592
//#include <cv.h>
//#include <cxcore.h>
#include <opencv2/opencv.hpp>
//#include "svlBase.h"
//#include "svlVision.h"

#define DRWN_EPSILON 1.0e-6

using namespace std;

// drwnHOGNormalization -----------------------------------------------------
//! HOG normalization method.
IplImage* drwnGreyImage(const IplImage* src);
// Pad image and copy boundary
IplImage *drwnPadImage(IplImage *src, int margin);
IplImage *drwnPadImage(const IplImage *src, int margin);
IplImage *drwnPadImage(IplImage *src, const CvRect& page);
IplImage *drwnPadImage(const IplImage *src, const CvRect& page);
void releaseOpenCVMatrices(vector<CvMat *>& matrices);
void releaseOpenCVMatrices(pair<CvMat *, CvMat *>& matrices);

typedef enum _drwnHOGNormalization {
    DRWN_HOG_L2_NORM, /*!< \f$ \frac{v}{\|v\|_2} \f$ */ 
    DRWN_HOG_L1_NORM, /*!< \f$ \frac{v}{\|v\|_1} \f$ */
    DRWN_HOG_L1_SQRT /*!< \f$ \sqrt{\frac{v}{\|v\|_1}} \f$ */
} drwnHOGNormalization;

// drwnHOGFeatures ----------------------------------------------------------
//! Encapsulates histogram-of-gradient (HOG) feature computation.
//!
//! The HOG features are described in:
//!  \li N. Dalal and B. Triggs, "Histogram of Oriented Gradients for Human
//!      Detection," CVPR 2005.
//!  \li Felzenszwalb et al., "Object Detection with Discriminitively Trained
//!      Part Based Model," PAMI 2010.
//!
//! The class can also be used for dense feature HOG calculation.

class drwnHOGFeatures {
 public:
    static int DEFAULT_CELL_SIZE;    //!< default cell size (in pixels)
    static int DEFAULT_BLOCK_SIZE;   //!< default block size (in cells)
    static int DEFAULT_BLOCK_STEP;   //!< default block increment (in cells)
    static int DEFAULT_ORIENTATIONS; //!< default number of quantized orientations
    static drwnHOGNormalization DEFAULT_NORMALIZATION; //!< default normalization method
    static double DEFAULT_CLIPPING_LB; //!< default lower-bound clipping in [0, 1)
    static double DEFAULT_CLIPPING_UB; //!< default upper-bound clipping in (0, 1]
    static bool DEFAULT_DIM_REDUCTION; //!< true for analytic dimensionality reduction

 protected:
    int _cellSize;        //!< size of each cell in pixels
    int _blockSize;       //!< number of cells in a block
    int _blockStep;       //!< step to next block in cells
    int _numOrientations; //!< number of orientations in histogram
    drwnHOGNormalization _normalization; //!< normalization method
    pair<double, double> _clipping;      //!< clipping for renormalization (0.0, 1.0 means none)
    bool _bDimReduction;  //!< use dimensionality reduction trick of Felzenszwalb et al, PAMI 2010

 public:
    drwnHOGFeatures();
    virtual ~drwnHOGFeatures();

    //! returns the number of features (\ref numBlocks times \ref _numOrientations)
    inline int numFeatures() const;
    //! returns the size of the feature maps in terms of cells
    inline CvSize numCells(const CvSize& imgSize) const;
    //! returns the size of the feature maps in terms of blocks
    inline CvSize numBlocks(const CvSize& imgSize) const;
    //! returns the size of the padded (enlarged) image over which features are computed
    inline CvSize padImageSize(const CvSize& imgSize) const;

    //! pre-process gradient magnitude and orientation (can be provided to \ref computeFeatures)
    pair<CvMat *, CvMat *> gradientMagnitudeAndOrientation(const IplImage *img) const;

    //! feature calculation from greyscale image
    //! returns features as a vector of matrices of size \ref numBlocks
    void computeFeatures(const IplImage *img, std::vector<CvMat *> &features);
    //! features calculation from gradient magnitude and orientation (returned by
    //! \ref gradientMagnitudeAndOrientation)
    void computeFeatures(const pair<CvMat *, CvMat *>& gradMagAndOri,
        std::vector<CvMat *> &features);

    //! compute features at each pixel location
    //! returns features as a vector of images the same size as the original image
    void computeDenseFeatures(const IplImage *img, std::vector<IplImage *> &features);

    //! visualization
    IplImage *visualizeCells(const IplImage *img, int scale = 2);
    //IplImage *visualizeBlocks(const IplImage *img, int scale = 1);

 protected:
    void computeCanonicalOrientations(vector<float>& x, vector<float>& y) const;
    void computeCellHistograms(const pair<CvMat *, CvMat *>& gradMagAndOri,
        vector<CvMat *>& cellHistorgams) const;
    void computeBlockFeatures(const vector<CvMat *>& cellHistograms, vector<CvMat *>& features) const;
    void normalizeFeatureVectors(std::vector<CvMat *>& features) const;
    void clipFeatureVectors(std::vector<CvMat *>& features) const;
};

// drwnHOGFeatures inline functions -----------------------------------------

inline int drwnHOGFeatures::numFeatures() const
{
    return _bDimReduction ? (_blockSize * _blockSize + _numOrientations) :
        (_blockSize * _blockSize * _numOrientations);
}

inline CvSize drwnHOGFeatures::numCells(const CvSize& imgSize) const
{
    return cvSize((int)((imgSize.width + _cellSize - 1) / _cellSize),
        (int)((imgSize.height + _cellSize - 1) / _cellSize));
}

inline CvSize drwnHOGFeatures::numBlocks(const CvSize& imgSize) const
{
    CvSize nCells = numCells(imgSize);
    return cvSize(nCells.width - _blockSize + 1, nCells.height - _blockSize + 1);
}

inline CvSize drwnHOGFeatures::padImageSize(const CvSize& imgSize) const
{
    return cvSize((int)((imgSize.width + _cellSize - 1) / _cellSize) * _cellSize,
        (int)((imgSize.height + _cellSize - 1) / _cellSize) * _cellSize);
}

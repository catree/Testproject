//
//  main.cpp
//  Test
//
//  Created by Marcel Tuchner on 22.09.16.
//  Copyright © 2016 Marcel Tuchner. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

typedef float sift_wt;

//default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

//determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

//determines the radius of the region used in oreintation assignment
static const float SIFT_ORI_RADIUS = 2.5 * SIFT_ORI_SIG_FCTR;

//orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

//default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = 3; //3 original opencv = 4

//default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = 4; //4 original opencv = 8

//determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

//threshold on magnitude of elments of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

//factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

//OutputArray _descriptors;
Mat src, dst, tmp, hist, bas, patch, patch5_5;
int rows, cols;

int descriptorSize() {
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}


static float calcOrientationHist(const Mat& img, Point pt, int radius, float sigma, float* hist, int n){
    
    int i, j, k, len = (radius*2+1)*(radius*2+1);
    
    float expf_scale = -1.f/(2.f * sigma * sigma);
    
    AutoBuffer<float> buf(len*4+n+4);
    
    float *X = buf, *Y = X + len, *Mag = X, *Ori = Y + len, *W = Ori + len;
    float* temphist = W + len + 2;
    
    for( i = 0; i < n; i++){
        
        temphist[i] = 0.f;
    }
    
    for(i = -radius, k = 0; i <= radius; i++){
        
        int y = pt.y+i;
        if( y <= 0 || y >= img.rows -1)
            continue;
        
        for(j = - radius; j <= radius; j++){
            
            int x = pt.x + j;
            if(x <= 0 || x >= img.cols -1)
                continue;
            float dx = (float)(img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y,x-1));
            float dy = (float)(img.at<sift_wt>(y-1,x) - img.at<sift_wt>(y+1,x));
            
            X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*expf_scale;
            k++;
        }
    }
    
    len = k;
    
    //compute gradient values, orientations and the weights over the pixel neighborhood
    hal::exp32f(W, W, len);
    hal::fastAtan2(Y, X, Ori, len, true);
    hal::magnitude32f(X, Y, Mag, len);
    
    for( k = 0; k < len; k ++){
        int bin = cvRound((n/360.f)*Ori[k]);
        if(bin >= n)
            bin -=n;
        if(bin < 0 )
            bin +=n;
        temphist[bin] += W[k]*Mag[k];
    }
    
    //smooth the histogram
    temphist[-1] = temphist[n-1];
    temphist[-2] = temphist[n-2];
    temphist[n] = temphist[0];
    temphist[n+1] = temphist[1];
    
    for(i = 0; i < n; i++){
        
        hist[i] = (temphist[i-2] + temphist[i+2])*(1.f/16.f)+
        (temphist[i-1] +temphist[i+1])*(4.f/16.f)+
        temphist[i]*(6.f/16.f);
    }
    
    float maxval = hist[0];
    for(i = 1; i < n; i++)
        maxval = std::max(maxval, hist[i]);
    
    return maxval;
}


static void calcSiftDescriptor(const Mat& img, Point2f ptf, float ori, float scl, int d, int n, float* dst){
    
    Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float expf_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    
    //clip the radius to the diagonal of the image to avoid autobuffer too large exception
    radius = std::min(radius, (int) sqrt(((double) img.cols)*img.cols + ((double) img.rows) * img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;
    
    int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (d+2)*(d+2)*(n+2);
    int rows = img.rows, cols = img.cols;
    
    AutoBuffer<float> buf(len*6 + histlen);
    
    float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
    float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;
    
    for(i = 0; i < d+2; i++){
        for(j = 0; j < d+2; j++){
            for(k = 0; k < n+2; k++){
                hist[(i * ( d + 2) + j) * (n + 2) + k] = 0.;
            }
        }
    }
    
    for( i = -radius, k=0; i <= radius; i++){
        for(j = -radius; j <= radius; j++){
            //Calculate sample's histogram array coords rotated relative to ori
            // Substract 0.5 so samples that fall e.g.in the center of row 1 (i.e.
            // r_rot = 1.5) have full weight placed in row 1 after interpolation.
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;
            
            if(rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 && r < rows-1 & c > 0 && c < cols -1){
                
                float dx = (float)(img.at<sift_wt>(r,c+1) - img.at<sift_wt>(r, c-1));
                float dy = (float)(img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1,c));
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot) * expf_scale;
                k++;
            }
        }
    }
    len = k;
    hal::fastAtan2(Y, X, Ori, len, true);
    hal::magnitude(X, Y, Mag, len);
    hal::exp32f(W, W, len);
    
    for(k = 0; k < len; k++){
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori) * bins_per_rad;
        float mag = Mag[k] * W[k];
        
        int r0 = cvFloor(rbin);
        int c0 = cvFloor(cbin);
        int o0 = cvFloor(obin);
        rbin -= r0;
        cbin -= c0;
        obin -= o0;
        
        if(o0 < 0)
            o0 += n;
        if(o0 >=n)
            o0 -=n;
        
        //histogram update using tri-linear interpolation
        float v_r1 = mag * rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1 * cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0 * cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11 * obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10 * obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01 * obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00 * obin, v_rco000 = v_rc00 - v_rco001;
        
        int idx = ((r0 +1)*(d+2) + c0+1)*(n+2) +o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }
    
    //finalize histogram, since the orientation histograms are circular
    for(i = 0; i < d; i++){
        for(j = 0; j < d; j++){
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for(k = 0; k < n; k++){
                dst[(i*d + j)*n + k] = hist[idx+k];       //EXC_BAD_ACCESS (CODE=1)
            }
        }
    }
    
    //copy histogram to the descriptor,
    //apply hysteresis thresholding
    //and scale the result, so that it can be easily converted
    //to byte array
    
    float nrm2 = 0;
    len = d*d*n;
    for(k = 0; k < len; k++)
        nrm2 += dst[k]*dst[k];
    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    for(i = 0, nrm2 = 0; i < k; i++){
        float val = std::min(dst[i], thr);
        dst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);
    
    for( k = 0; k < len; k++){
        
        dst[k] = saturate_cast<uchar>(dst[k]*nrm2);
    }
}

static void calcDescriptors(const vector<Mat>& pyr, const vector<KeyPoint>& features, Mat& descriptors ){
    
    int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
    
    for(size_t i = 0; i < features.size(); i++){
        
        KeyPoint kpt = features[i];
        
        float scale = 1.f/(1 << kpt.octave);
        float size = kpt.size * scale;
        Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);
        const Mat& img  = pyr[kpt.octave];
        
        float angle = 360.f - kpt.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;
        calcSiftDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));
    }
    
}

int main(int argc, const char * argv[]) {
    
    const int n = SIFT_ORI_HIST_BINS;
    float hist [n];
    
    vector<vector<KeyPoint>> candidates2d;
    vector<vector<KeyPoint>> features2d;
    vector<KeyPoint> candidates;
    vector<KeyPoint> features;
    vector<Mat> Pyr;
    String imgName("/users/mtuchner/desktop/test/test/example.jpg");
    bas = imread(imgName, -1);
    int threshold = 44;
    int substract = 14;
    int l = 0;
    cvtColor(bas, src, CV_BGR2GRAY);
    
    Pyr.push_back(src);
    double factor = sqrt(2);
    int k = 1;
    do{
        if(candidates.size() > 0){
            candidates.clear();
        }
        rows = src.rows/factor;
        cols = src.cols/factor;
        
        resize(src, tmp, Size(rows, cols),0,0,INTER_NEAREST);
        src.release();
        
        GaussianBlur(tmp, dst, Size(5,5),0,0);
        tmp.release();
        
        FAST(dst, candidates, threshold, true);
        
        Pyr.push_back(dst);
        src = dst.clone();
        dst.release();
        
        if(candidates.size() > 100){
            
            for(int i = 0; i < candidates.size(); i++){
                if(candidates[i].pt.x > 7 && candidates[i].pt.y > 7){
                    KeyPoint kpt = candidates[i];
                    
                    //Really necessary?
                    //int px = kpt.pt.x -7;
                    //int py = kpt.pt.y -7;
                    //if(px + 15 >= src.cols || py + 15 >= src.rows)
                    //    continue;
                    
                    //patch = src(Rect(px,py,15,15));
                    //Point centre(7,7);

                    float scl_octv = kpt.size * 0.5f / (1 << k);
                    
                    float omax = calcOrientationHist(src, kpt.pt, cvRound(SIFT_ORI_RADIUS * scl_octv), SIFT_ORI_SIG_FCTR * scl_octv, hist, n);
                    
                    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
            
                    for(int j = 0; j < n; j++){
                        int l = j > 0 ? j - 1 : n -1;
                        int r2 = j < n-1 ? j+1 : 0;
                
                        if( hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr ) {
                            float bin = j + 0.5f * ( hist[i] - hist[r2] ) / ( hist[l] - 2 * hist[j] + hist[r2]);
                            bin = bin < 0 ? n +bin : bin >= n ? bin - n : bin;
                            kpt.angle = 360.f - (float)((360.f/n) * bin);
                            if(std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                kpt.angle = 0.f;
                            kpt.octave = k;
                            features.push_back(kpt);
                        }
                    }
                    
                }
                
                //patch.release();
            }
            k++;
            if(threshold > 0){
                 threshold -= substract;
                if(threshold < 0){
                    threshold = 0;
                }
                if(substract > 0){
                    substract -=4;
                    if(substract <= 0){
                        substract = 6 + l;
                        l++;
                    }
                }
            }
        }
    }while(candidates.size() > 100);
    
    //remove duplicated
    KeyPointsFilter::removeDuplicated(features);
    
    int dsize = descriptorSize();
    
    //_descriptors.create((int)features.size(), dsize, CV_32F);
    Mat descriptors((int)features.size(), dsize, CV_32F);
    
    calcDescriptors(Pyr, features, descriptors);
    
    
    /*
    k = 1;
    for(int j = 0; j < features2d.size(); j++){
        cout << "octave: " << k << endl;
        cout << "Feature Size: " << features2d[j].size() << endl;
        k++;
        for(int i = 0; i < features2d[j].size(); i ++){
            cout << "FeaturePoint: " << features2d[j][i].pt << endl;
        }
    }
    
    k = 1;
    for(int j = 0; j < candidates2d.size(); j++){
        cout << "octave: " << k << endl;
        cout << "Candidate Size: " << candidates2d[j].size() << endl;
        k++;
    

    }
    */
    waitKey();


    return 0;
}

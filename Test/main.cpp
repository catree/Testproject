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

static const int SIFT_ORI_HIST_BINS = 36;
static const float SIFT_ORI_SIG_FCTR = 1.5f;
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

Mat src; Mat dst; Mat tmp; Mat hist; Mat patch; Mat bas; Mat patch5_5;
int rows, cols;

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
            //in opencv sift at<sift_wt>
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
        int bin = cvRound((n/360.f)+Ori[k]);
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

int main(int argc, const char * argv[]) {
    
    const int n = SIFT_ORI_HIST_BINS;
    float hist [n];
    
    vector<vector<KeyPoint>> candidates2d;
    vector<vector<KeyPoint>> features2d;
    vector<KeyPoint> candidates;
    vector<KeyPoint> features;
    vector<Mat> Pyr;
    String imgName("/users/mtuchner/desktop/test/test/example.jpg");
    bool gotKeypoint = false;
    bas = imread(imgName, -1);
    int threshold = 55;
    int substract = 16;
    int l = 0;
    cvtColor(bas, src, CV_BGR2GRAY);
    
    Pyr.push_back(src);
    double factor = sqrt(2);
    int k = 1;
    do{
        
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
            
//for(int i = 0; i < candidates.size(); i++){
//   candidates[i].octave = k;
//}
            for(int i = 0; i < candidates.size(); i++){
                if(candidates[i].pt.x > 7 && candidates[i].pt.y > 7){
                    KeyPoint kpt = candidates[i];
                    
                    
                    int px = kpt.pt.x -7;
                    int py = kpt.pt.y -7;
                    if(px + 15 >= src.cols || py + 15 >= src.rows)
                        continue;
                    
                    patch = src(Rect(px,py,15,15));
                    Point centre(7,7);

                    float scl_octv = kpt.size * 0.5f / (1 << 1);
            
                    float omax = calcOrientationHist(patch, centre, cvRound(SIFT_ORI_RADIUS * scl_octv), SIFT_ORI_SIG_FCTR * scl_octv, hist, n);
                    
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
                            features.push_back(kpt);
                            gotKeypoint = true;
                        }
                    }
                   if(gotKeypoint){
                        for(int i = 0; i < patch.rows; i += 5 ){
                            for(int j = 0; j < patch.cols; j+= 5){
                            
                                patch5_5 = patch(Rect(j,i,5,5));
                                
                                
                                patch5_5.release();
                                gotKeypoint = false;
                            }
                        }
                   }
                }
                patch.release();
            }
            k++;
            if(threshold > 0){
                 threshold -= substract;
                if(threshold < 0){
                    threshold = 0;
                }
                if(substract > 0){
                    substract -=6;
                    if(substract <= 0){
                        substract = 5 + l;
                        l++;
                    }
                }
            }
            features2d.push_back(features);
            //features.clear();
            candidates2d.push_back(candidates);
            candidates.clear();
        }
    }while(candidates.size() > 100);

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
    
    waitKey();


    return 0;
}

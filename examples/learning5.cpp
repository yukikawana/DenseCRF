#include "densecrf.h"
#include "optimization.h"
#include <cstdio>
#include <cmath>	
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include "ppm.h"
#include "common.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <utility>
#include <algorithm>


//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
// The energy object implements an energy function that is minimized using LBFGS
class CRFEnergy: public EnergyFunction {
protected:
	VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
	DenseCRF2D & crf_;
	
	int NIT_;
	bool unary_, pairwise_, kernel_;
	float l2_norm_;
	//vector<pair<pair<VectorXf, VectorXf>,pair<const unsigned char *, IntersectionOverUnion> >> data;
	vector<pair<DenseCRF2D, IntersectionOverUnion> > crfs;
public:

	CRFEnergy( DenseCRF2D & crf, vector<pair<DenseCRF2D, IntersectionOverUnion> > & ocrfs, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crf_(crf),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f), crfs(ocrfs){
		initial_u_param_ = crf_.unaryParameters();
		initial_lbl_param_ = crf_.labelCompatibilityParameters();
		initial_knl_param_ = crf_.kernelParameters();

	}
	void setL2Norm( float norm ) {
		l2_norm_ = norm;
	}
	virtual VectorXf initialValue() {
		VectorXf p( unary_*initial_u_param_.rows() + pairwise_*initial_lbl_param_.rows() + kernel_*initial_knl_param_.rows() );
		p << (unary_?initial_u_param_:VectorXf()), (pairwise_?initial_lbl_param_:VectorXf()), (kernel_?initial_knl_param_:VectorXf());
		return p;
	}
	virtual double gradient( const VectorXf & x, VectorXf & dx ) {
		VectorXf du = 0*initial_u_param_, dl = 0*initial_lbl_param_, dk = 0*initial_knl_param_;
		double r2(0);
		//cout<<"gradient in"<<endl;
		dx.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
		dx.fill(0.f);
		//cout<<"resize"<<endl;
		for(int i=0;i<crfs.size();i++)
		{
		int p = 0;
		if (unary_) {
			crfs[i].first.setUnaryParameters( x.segment( p, initial_u_param_.rows()));
//			crf_.setUnaryParameters( x.segment( p, initial_u_param_.rows()));
			p += initial_u_param_.rows();
		}
		
		if (pairwise_) {
			crfs[i].first.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
//			crf_.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
			p += initial_lbl_param_.rows();
		}
		if (kernel_)
			crfs[i].first.setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );
//			crf_.setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );
		
		//double r = crf_.gradient( NIT_, crfs[i].second, unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );
		double r = crfs[i].first.gradient( NIT_, crfs[i].second, unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );
		
		
		//cout<<"r="<<r<<endl;
		VectorXf dx2;
		dx2.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
		dx2.fill(0.f);
		dx2 << -(unary_?du:VectorXf()), -(pairwise_?dl:VectorXf()), -(kernel_?dk:VectorXf());
		dx+=dx2;
		//cout<<"grad dx "<<dx.size()<<endl;
		r2 -= r;
		}
		if( l2_norm_ > 0 ) {
			dx += l2_norm_ * x;
			r2 += 0.5*l2_norm_ * (x.dot(x));
		}
		cout<<"energy="<<r2<<endl;
		return r2;
	}
};

int main( int argc, char* argv[]){

	const String dir = "/home/kawana/workspace/msrc9/";
	int M = 6;
	MatrixXf logistic_transform( M, 4 );
	int NIT = 5;
	int W, H, GW, GH;
	GW = 320;
	GH = 213;
	W = GW;
	H = GH;
	
	for( int j=0; j<logistic_transform.cols(); j++ )
		for( int i=0; i<logistic_transform.rows(); i++ )
			logistic_transform(i,j) = 0.01*(1-2.*rand()/RAND_MAX);
	


	int iter =7;
	int batch = 2;
	int maxidx = 22;
	int minidx = 1;
	int idx = minidx; 
	float slr =  0.01;
	float alpha = -0.00001;
	float mu = 0.99;
	float eps = 0.00000001; 
	int pgweight =1;
	MatrixXf learning_params( 3,3 );
	// Optimize the CRF in 3 phases:
	//  * First unary only
	//  * Unary and pairwise
	//  * Full CRF

	learning_params<<1,0,0,
	                 1,1,0,
					 1,1,1;
	//learning_params<<1,0,0;
					// 1,1,0;
	vector<VectorXf> param;
	bool first = true;

	ostringstream ostgt;
	ostgt << "1_" << idx << "_s.bmp";
	const String filename2 = dir+ostgt.str();
	Mat y =	imread(filename2);
	MatrixXf ey;
	//cv2eigen(y.reshape(1,y.rows*y.cols),ey);
	unsigned char * anno = y.data;
	const int N2 = W*H;
	MatrixXf logistic_feature2( 4, N2 );
	DenseCRF2D crf_fi(W, H, M);


	cout<<"dd"<<endl;
	crf_fi.setUnaryEnergy( logistic_transform, logistic_feature2 );
	cout<<"dd"<<endl;
	crf_fi.addPairwiseGaussian( 3, 3, new PottsCompatibility( pgweight ) );
	cout<<"dd"<<endl;
	crf_fi.addPairwiseBilateral( 80, 80, 13, 13, 13, anno, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	cout<<"dd"<<endl;

	VectorXf iniparam;
	cout<<"crf_fi initial param"<<crf_fi.unaryParameters().transpose()<<crf_fi.labelCompatibilityParameters().transpose()<<crf_fi.kernelParameters().transpose()<<endl;
	cout<<"dd"<<endl;


	vector<int> ridx;
	for(int i = minidx;i<maxidx+1;i++){
		if(i != 17)
			ridx.push_back(i);
	}
		vector<pair<DenseCRF2D, IntersectionOverUnion> > ocrfs;
	for(int b = 0;b < ridx.size(); b++){
	idx = b;	
	cout << "idx=" << ridx.at(idx) << endl;
	ostringstream osts;
	ostringstream ostgt;
	osts << "1_" << ridx.at(idx) << "_s.bmp";
	ostgt << "1_" << ridx.at(idx) << "_s_GT.bmp";
	String filename = dir+osts.str();
	Mat x = imread(filename);
	const String filename2 = dir+ostgt.str();
	Mat y =	imread(filename2);


	const int N = W*H;
	cout<<"set enegy"<<endl;
	MatrixXf logistic_feature( 4, N );
	logistic_feature.fill( 1.f );
	for( int i=0; i<N; i++ )
		for( int k=0; k<3; k++ ){
			logistic_feature(k,i) = x.data[3*i+k] / 255.;
		}
	// Setup the CRF model
	VectorXs labeling = crf_fi.getLabeling( y.data, W*H, M );
	DenseCRF2D crf(W, H, M);
	IntersectionOverUnion objective( labeling );
	crf.setUnaryEnergy( logistic_transform, logistic_feature );
	crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( pgweight ) );
	crf.addPairwiseBilateral( 80, 80, 13, 13, 13,  x.data, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	ocrfs.push_back(make_pair(crf,objective));
	}


	for( int ip=0; ip<learning_params.rows(); ip++ )
	{
	CRFEnergy energy( crf_fi,ocrfs, NIT, learning_params(ip,0), learning_params(ip,1), learning_params(ip,2) );
	energy.setL2Norm( 1e-3 );
//	energy.addCRFS(ocrfs);
	
	
	//random_shuffle(ridx.begin(),ridx.end());

	float lossb=0;
	
	
		cout<<"minimize"<<endl;

		VectorXf p = minimizeLBFGS( energy, 2, true );
				cout<<"minimize done"<<endl;
		int id = 0;
		if( learning_params(ip,0) ) {
			crf_fi.setUnaryParameters( p.segment( id, crf_fi.unaryParameters().rows() ) );
			id += crf_fi.unaryParameters().rows();
		cout<<"end 1"<<endl;
		}
		if( learning_params(ip,1) ) {
			crf_fi.setLabelCompatibilityParameters( p.segment( id, crf_fi.labelCompatibilityParameters().rows() ) );
			id += crf_fi.labelCompatibilityParameters().rows();
		cout<<"end 2"<<endl;
		}
		if( learning_params(ip,2) )
			crf_fi.setKernelParameters( p.segment( id, crf_fi.kernelParameters().rows() ) );		

		}

	cout<<"learnt weight"<<crf_fi.unaryParameters().transpose()<<" " <<crf_fi.labelCompatibilityParameters().transpose() <<" "<<crf_fi.kernelParameters().transpose()<<endl;
	
	idx = 17;
	ostringstream osts;
	osts << "1_" << idx << "_s.bmp";
	String filename = dir+osts.str();
	Mat x = imread(filename);
	MatrixXf ex;
	cv2eigen(x.reshape(1,x.rows*x.cols),ex);
	unsigned char * im2 = x.data;
	const int N = W*H;


	MatrixXf logistic_feature( 4, N );
	logistic_feature.fill( 1.f );
	for( int i=0; i<N; i++ )
		for( int k=0; k<3; k++ )
			logistic_feature(k,i) = x.data[3*i+k] / 255.;
		DenseCRF2D crf(W, H, M);
	crf.setUnaryEnergy( logistic_transform, logistic_feature );	
	crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( pgweight ) );
	crf.addPairwiseBilateral( 80, 80, 13, 13, 13, x.data, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	cout<<"initial inference weight"<<crf.unaryParameters().transpose()<<" " <<crf.labelCompatibilityParameters().transpose() <<" "<<crf.kernelParameters().transpose()<<endl;
	
	crf.setUnaryParameters(crf_fi.unaryParameters());
	crf.setLabelCompatibilityParameters(crf_fi.labelCompatibilityParameters());
	crf.setKernelParameters( crf_fi.kernelParameters());
	cout<<"inference weight"<<crf.unaryParameters().transpose()<<" " <<crf.labelCompatibilityParameters().transpose() <<" "<<crf.kernelParameters().transpose()<<endl;
	NIT=10;
	VectorXs map = crf.map(NIT);
	
	// Store the result
	unsigned char *res = crf_fi.colorize( map, W, H );
	//unsigned char *res = colorize( map, W, H );

	Mat out = Mat(H,W,CV_8UC3,res);
	imwrite("out2.bmp",out);
	imshow("out",out);
	imshow("x",x);
	waitKey(0);
	//writePPM( argv[3], W, H, res );
	delete[] im2;
	delete[] res;
}

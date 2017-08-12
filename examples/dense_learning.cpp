/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
//#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
// The energy object implements an energy function that is minimized using LBFGS
class CRFEnergy: public EnergyFunction {
protected:
	VectorXf initial_u_param_, initial_lbl_param_, initial_knl_param_;
	DenseCRF & crf_;
	const ObjectiveFunction & objective_;
	int NIT_;
	bool unary_, pairwise_, kernel_;
	float l2_norm_;
public:
	CRFEnergy( DenseCRF & crf, const ObjectiveFunction & objective, int NIT, bool unary=1, bool pairwise=1, bool kernel=1 ):crf_(crf),objective_(objective),NIT_(NIT),unary_(unary),pairwise_(pairwise),kernel_(kernel),l2_norm_(0.f){
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
		int p = 0;
		if (unary_) {
			crf_.setUnaryParameters( x.segment( p, initial_u_param_.rows() ) );
			p += initial_u_param_.rows();
		}
		if (pairwise_) {
			crf_.setLabelCompatibilityParameters( x.segment( p, initial_lbl_param_.rows() ) );
			p += initial_lbl_param_.rows();
		}
		if (kernel_)
			crf_.setKernelParameters( x.segment( p, initial_knl_param_.rows() ) );
		
		VectorXf du = 0*initial_u_param_, dl = 0*initial_u_param_, dk = 0*initial_knl_param_;
		double r = crf_.gradient( NIT_, objective_, unary_?&du:NULL, pairwise_?&dl:NULL, kernel_?&dk:NULL );
		dx.resize( unary_*du.rows() + pairwise_*dl.rows() + kernel_*dk.rows() );
		dx << -(unary_?du:VectorXf()), -(pairwise_?dl:VectorXf()), -(kernel_?dk:VectorXf());
		r = -r;
		if( l2_norm_ > 0 ) {
			dx += l2_norm_ * x;
			r += 0.5*l2_norm_ * (x.dot(x));
		}
		
		return r;
	}
};

int main( int argc, char* argv[]){

	const String dir = "C:\\Users\\User\\Downloads\\msrc21\\";
	const int M = 5;
	MatrixXf logistic_transform( M, 4 );

	int W, H, GW, GH;
	GW = 320;
	GH = 213;
	W = GW;
	H = GH;
	DenseCRF2D crf(W, H, M);
	for( int j=0; j<logistic_transform.cols(); j++ )
		for( int i=0; i<logistic_transform.rows(); i++ )
			logistic_transform(i,j) = 0.01*(1-2.*rand()/RAND_MAX);
	


	int idx = 17;
	for(idx = 17;idx < 18; idx++){
		cout << "idx=" << idx << endl;
	ostringstream osts;
	ostringstream ostgt;
	osts << "1_" << idx << "_s.bmp";
	ostgt << "1_" << idx << "_s_GT.bmp";
	String filename = dir+osts.str();
	//String filename = "../im1.jpeg";
	Mat x = imread(filename);
	//x.convertTo(x,CV_8UC3);
	const String filename2 = dir+ostgt.str();
	//String filename2 = "../anno1.jpeg";
	Mat y =	imread(filename2);

	//y.convertTo(y,CV_8UC3);
	MatrixXf ey;
	cv2eigen(y.reshape(1,y.rows*y.cols),ey);
	MatrixXf ex;
	cv2eigen(x.reshape(1,x.rows*x.cols),ex);



	// Number of labels [use only 4 to make our lives a bit easier]


	
	// Load the color image and some crude annotations (which are used in a simple classifier)

	unsigned char * im = x.data;
	unsigned char * anno = y.data;

	VectorXs labeling = getLabeling( anno, W*H, M );
	const int N = W*H;
	
	// Get the logistic features (unary term)
	// Here we just use the color as a feature
	MatrixXf logistic_feature( 4, N );
	logistic_feature.fill( 1.f );
	for( int i=0; i<N; i++ )
		for( int k=0; k<3; k++ )
			logistic_feature(k,i) = im[3*i+k] / 255.;
	
	// Setup the CRF model

	// Add a logistic unary term
	crf.setUnaryEnergy( logistic_transform, logistic_feature );
	
	// Add simple pairwise potts terms
	crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 1 ) );
	// Add a longer range label compatibility term
	crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	
	// Choose your loss function
// 	LogLikelihood objective( labeling, 0.01 ); // Log likelihood loss
// 	Hamming objective( labeling, 0.0 ); // Global accuracy
// 	Hamming objective( labeling, 1.0 ); // Class average accuracy
// 	Hamming objective( labeling, 0.2 ); // Hamming loss close to intersection over union
	IntersectionOverUnion objective( labeling ); // Intersection over union accuracy
	
	int NIT = 5;
	const bool verbose = true;
	
	MatrixXf learning_params( 3, 3 );
	// Optimize the CRF in 3 phases:
	//  * First unary only
	//  * Unary and pairwise
	//  * Full CRF
	learning_params<<1,0,0,
	                 1,1,0,
					 1,1,1;
	
	for( int i=0; i<learning_params.rows(); i++ ) {
		// Setup the energy
		CRFEnergy energy( crf, objective, NIT, learning_params(i,0), learning_params(i,1), learning_params(i,2) );
		energy.setL2Norm( 1e-3 );
		cout<<"ini v"<<energy.initialValue().size()<<" "<<energy.initialValue().cols()<<" "<<energy.initialValue().rows()<<endl;
		// Minimize the energy

		
		VectorXf p = minimizeLBFGS( energy, 2, true );
		
		// Save the values
		int id = 0;
		if( learning_params(i,0) ) {
			crf.setUnaryParameters( p.segment( id, crf.unaryParameters().rows() ) );
			id += crf.unaryParameters().rows();
		}
		if( learning_params(i,1) ) {
			crf.setLabelCompatibilityParameters( p.segment( id, crf.labelCompatibilityParameters().rows() ) );
			id += crf.labelCompatibilityParameters().rows();
		}
		if( learning_params(i,2) )
			crf.setKernelParameters( p.segment( id, crf.kernelParameters().rows() ) );
			
	}
	// Return the parameters
	std::cout<<"Unary parameters: "<<crf.unaryParameters().transpose()<<std::endl;
	std::cout<<"Pairwise parameters: "<<crf.labelCompatibilityParameters().transpose()<<std::endl;
	std::cout<<"Kernel parameters: "<<crf.kernelParameters().transpose()<<std::endl;
	
	// Do map inference

	/*
	VectorXs map = crf.map(NIT);
	
	// Store the result
	unsigned char *res = colorize( map, W, H );

	Mat out = Mat(H,W,CV_8UC3,res);

	//imshow("out",out);
	//waitKey(0);
	//imwrite("out2.bmp",out);
	//writePPM( argv[3], W, H, res );
	
	delete[] im;
	delete[] anno;
	delete[] res;
	*/		
	}
	idx = 17;
	ostringstream osts;
	osts << "1_" << idx << "_s.bmp";
	String filename = dir+osts.str();
	Mat x = imread(filename);
	MatrixXf ex;
	cv2eigen(x.reshape(1,x.rows*x.cols),ex);
	unsigned char * im = x.data;
	const int N = W*H;
	

	MatrixXf logistic_feature( 4, N );
	logistic_feature.fill( 1.f );
	for( int i=0; i<N; i++ )
		for( int k=0; k<3; k++ )
			logistic_feature(k,i) = im[3*i+k] / 255.;
	
	//crf.setUnaryEnergy( logistic_transform, logistic_feature );	
	//// Add simple pairwise potts terms
	//crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 1 ) );
	//// Add a longer range label compatibility term
	//crf.addPairwiseBilateral( 80, 80, 13, 13, 13, im, new MatrixCompatibility( MatrixXf::Identity(M,M) ) );
	
	int NIT = 5;
	VectorXs map = crf.map(NIT);
	
	// Store the result
	unsigned char *res = colorize( map, W, H );

	Mat out = Mat(H,W,CV_8UC3,res);
	imshow("out",out);
	waitKey(0);
	imwrite("out2.bmp",out);
	writePPM( argv[3], W, H, res );
	delete[] im;
	delete[] res;
}

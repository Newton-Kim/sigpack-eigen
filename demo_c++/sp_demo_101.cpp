#include "sigpack.h"

#include <iostream>

using Eigen::VectorXd;
using namespace sp;

int main()
{
	// Filter coeffs.
	VectorXd b;
	int N = 15;

	// Create a FIR filter
	FIR_filt<double,double,double> fir_filt;
	b = fir1(7,0.35);
	fir_filt.set_coeffs(b);

	VectorXd X = VectorXd::Zero(N);  // Input sig
	X[0] = 1;  // Impulse 

	VectorXd Y = VectorXd::Zero(N);  // Output sig
	
	// Filter - sample loop
	for(int n=0;n<N;n++)
	{
		Y[n] = fir_filt(X[n]);
	}

	std::cout << "Filter coeffs: \n" << b.transpose() << std::endl;
	std::cout << "Impulse response:\n" << Y.transpose() << std::endl;

	// Create a IIR filter
	IIR_filt<double,double,double> iir_filt;
	VectorXd a;
	b << 0.25 , 0.5 , 0.25;
	a << 1.0 , -0.9;

	iir_filt.set_coeffs(b,a);
	Y = iir_filt.filter(X);
	std::cout << "IIR theor. impulse response: \n   0.2500   0.7250   0.9025   0.8123   0.7310   0.6579   0.5921   0.5329   0.4796   0.4317   0.3885 ..." << std::endl;
	std::cout << "IIR output: \n" << Y.transpose() << std::endl;

	Delay<double> del(3);
	std::cout << "Delay three samples:\n" << del.delay(Y).transpose() << std::endl;
	
	return 1;
}

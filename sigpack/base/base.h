// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SP_BASE_H
#define SP_BASE_H

namespace sp
{
    ///
    /// @defgroup math Math
    /// \brief Math functions.
    /// @{

    const double PI   = 3.14159265358979323846;     ///< ... _or use arma::datum::pi_
    const double PI_2 = 6.28318530717958647692;

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief A sinc, sin(x)/x, function.
    /// @param x The angle in radians
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE double sinc( double x )
    {
        if(x==0.0)
            return 1.0;
        else
            return std::sin(PI*x)/(PI*x);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief A sinc, sin(x)/x, function.
    /// @param x The angle in radians
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd sinc(const Eigen::VectorXd& x)
    {
        Eigen::VectorXd out;
        out.resize(x.size());
        for (unsigned int n = 0; n < out.size(); n++)
        {
            out(n) = sinc(x(n));
        }
        return out;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Modified first kind bessel function order zero.
    ///
    /// See bessel functions on [Wikipedia](https://en.wikipedia.org/wiki/Bessel_function)
    /// @param x
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE double besseli0( double x )
    {
        double y=1.0,s=1.0,x2=x*x,n=1.0;
        while (s > y*1.0e-9)
        {
            s *= x2/4.0/(n*n);
            y += s;
            n += 1;
        }
        return y;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Calculates angle in radians for complex input.
    /// @param x Complex input value
    ////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    double angle( const std::complex<T>& x )
    {
        return std::arg(x);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Calculates angle in radians for complex input.
    /// @param x Complex input vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd angle( const Eigen::VectorXcd& x )
    {
        Eigen::VectorXd P;
        P.resize(x.size());
        for(unsigned int r=0; r<x.rows(); r++)
            P(r) = std::arg(x(r));
        return P;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Calculates angle in radians for complex input.
    /// @param x Complex input matrix
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::MatrixXd angle( const Eigen::MatrixXcd& x )
    {
        Eigen::MatrixXd P;
        P.resize(x.rows(), x.cols());
        for(unsigned int r=0; r<x.rows(); r++)
            for(unsigned int c=0; c<x.cols(); c++)
                P(r,c) = std::arg(x(r,c));
        return P;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Unwraps the angle vector x, accumulates phase.
    /// @param x Complex input vector
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd unwrap( const Eigen::VectorXd& x )
    {
        Eigen::VectorXd P;
        double pacc = 0, pdiff = 0;
        const double thr=PI*170/180;
        P.resize(x.size());
        P(0)=x(0);
        for(unsigned int r=1; r<x.rows(); r++)
        {
            pdiff = x(r)-x(r-1);
            if( pdiff >= thr ) pacc += -PI_2;
            if( pdiff <= -thr) pacc +=  PI_2;
            P(r)=pacc+x(r);
        }
        return P;
    }
    /// @} // END math


    ///
    /// @defgroup data Data
    /// \brief Data generation/manipulation ...
    /// @{

    /// \brief Generates a linear time vector with specified sample rate. Delta time=1/Fs.
    /// @param N  Number of data points
    /// @param Fs Sample rate
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd timevec( const int N, const double Fs )
    {
        return Eigen::VectorXd::LinSpaced(1,0,N-1.0)/Fs;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief 1D FFT shift.
    /// @returns Circular shifted FFT
    /// @param Pxx Complex FFT
    ////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    Eigen::Vector<T, Eigen::Dynamic> fftshift(const Eigen::Vector<T, Eigen::Dynamic>& Pxx)
    {
        Eigen::Vector<T, Pxx.n_elem> x;
        x = shift(Pxx, floor(Pxx.n_elem / 2));
        return x;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief 1D FFT inverse/reverse shift.
    /// @returns Circular shifted FFT
    /// @param Pxx Complex FFT
    ////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    Eigen::Vector<T, Eigen::Dynamic> ifftshift(const Eigen::Vector<T, Eigen::Dynamic>& Pxx)
    {
        Eigen::Vector<T, Pxx.n_elem> x;
        x = shift(Pxx, -ceil(Pxx.n_elem / 2));
        return x;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief 2D FFT shift.
    /// @returns Circular shifted FFT
    /// @param Pxx FFT
    ////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fftshift(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Pxx)
    {
        uword R = Pxx.rows();
        uword C = Pxx.cols();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x;
	x.resize(R, C);
        x = shift(Pxx, static_cast<sword>(floor(R / 2)), 0);
        x = shift(x, static_cast<sword>(floor(C / 2)), 1);
        return x;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief 2D FFT inverse/reverse shift.
    /// @returns Circular shifted FFT
    /// @param Pxx FFT
    ////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ifftshift(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Pxx)
    {
        uword R = Pxx.rows();
        uword C = Pxx.cols();
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> x;
	x.resize(R, C);
        x = shift(Pxx, -ceil(R / 2), 0);
        x = shift(x, -ceil(C / 2), 1);
        return x;
    }

    /// @} // END data

    ///
    /// @defgroup misc Misc
    /// \brief Misc functions, error handling etc.
    /// @{

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief SigPack version string
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE std::string sp_version(void)
    {
        return std::to_string(SP_VERSION_MAJOR)+"."+std::to_string(SP_VERSION_MINOR)+"."+std::to_string(SP_VERSION_PATCH);
    }

    ///////////////////////////////////
    // err_handler("Error string")
    //      Prints an error message, waits for input and
    //      then exits with error
#define err_handler(msg) \
    { \
        std::cout << "SigPack Error [" << __FILE__  << "@" << __LINE__ << "]: " << msg << std::endl; \
        std::cin.get(); \
        exit(EXIT_FAILURE);\
    }

    ///////////////////////////////////
    // wrn_handler("Warning string")
    //      Prints an warning message
#define wrn_handler(msg)  \
    { \
        std::cout << "SigPack warning [" << __FILE__ << "@" << __LINE__ << "]: " << msg << std::endl;\
    }
    /// @} // END misc

} // end namespace
#endif


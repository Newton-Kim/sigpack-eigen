// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SP_FILTER_H
#define SP_FILTER_H
namespace sp
{
    ///
    /// @defgroup filter Filter
    /// \brief FIR/MA and IIR/ARMA filter functions.
    /// @{

    ///
    /// \brief FIR/MA filter class.
    ///
    /// Implements FIR/MA filter functions as \f[  y(n) = \sum_{k=0}^{M-1}{b_kx(n-k)}=b_0x(n)+b_1x(n-1)+...+b_{M-1}x(n-(M-1))\f]
    /// where M is the number of taps in the FIR filter. The filter order is M-1.
    /// Adaptive update of filter is possible with LMS or NLMS algorithms
    template <class T1, class T2, class T3>
    class FIR_filt
    {
    private:
        // Ordinary FIR filter
        uword M;           ///< Nr of filter taps
        uword cur_p;       ///< Pointer to current sample in buffer
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf;       ///< Signal buffer
        Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> b;         ///< Filter coefficients
        // Adaptive LMS FIR filter
        double mu;               ///< Adaptive filter step size
        uword L;           ///< Adaptive filter block length
        uword blk_ctr;     ///< Adaptive filter block length counter
        T2 c;                    ///< Adaptive filter NLMS regulation const.
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> P;         ///< Adaptive filter Inverse corr matrix (estimated accuracy)
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> Q;         ///< Adaptive filter Process noise
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> R;         ///< Adaptive filter Measurement noise
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> K;         ///< Adaptive filter gain vector
        double lmd;              ///< Adaptive filter RLS forgetting factor
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> X_tpz;     ///< Adaptive filter Toeplitz for Corr matrix calc.
        uword do_adapt;    ///< Adaptive filter enable flag
    public:
        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Constructor.
        ////////////////////////////////////////////////////////////////////////////////////////////
        FIR_filt(){}

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Destructor.
        ////////////////////////////////////////////////////////////////////////////////////////////
        ~FIR_filt(){}

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Clears the internal states and pointer.
        ////////////////////////////////////////////////////////////////////////////////////////////
        void clear(void)
        {
            buf = Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>::Zero( buf.rows(), buf.cols());
            cur_p = 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Sets coefficients in FIR filter.
        /// The internal state and pointers are cleared
        /// @param _b Filter coefficients \f$ [b_0 ..b_{M-1}]^T \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
        void set_coeffs(const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> &_b)
        {
            M = _b.n_elem;
            buf.resize(M,1);
            this->clear();
            b.resize(M,1);
            b = _b;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Sets coefficients in FIR filter (col format)
        /// The internal state and pointers are cleared
        /// @param _b_col Filter coefficients \f$ [b_0 ..b_{M-1}]^T \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
        void set_coeffs(const Eigen::Vector<T2, Eigen::Dynamic> &_b_col)
        {
          Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> b_mat = arma::conv_to<Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> >::from(_b_col);
          set_coeffs(b_mat);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Get coefficients from FIR filter.
        /// @return b Filter coefficients \f$ [b_0 ..b_{M-1}]^T \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
	Eigen::Vector<T2, Eigen::Dynamic> get_coeffs()
        {
           return arma::conv_to<Eigen::Vector<T2, Eigen::Dynamic> >::from(b);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Updates coefficients in FIR filter without clearing the internal states.
        /// @param _b Filter coefficients \f$ [b_0 ..b_{M-1}] \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
        void update_coeffs(const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> &_b)
        {
            b = _b;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Filter operator.
        /// @return Filtered output
        /// @param in Input sample
        ////////////////////////////////////////////////////////////////////////////////////////////
        T3 operator()(const T1 & in)
        {
            T3 out=0;
            uword p = 0;
            buf[cur_p] = in;                    // Insert new sample
            for( uword m = cur_p; m < M; m++)
                out += b[p++]*buf[m];           // Calc upper part
            for( uword m = 0; m < cur_p; m++)
                out += b[p++]*buf[m];           // ... and lower

            // Move insertion point
            if(cur_p == 0)
                cur_p = M-1;
            else
                cur_p--;

            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Filter function.
        /// @return Filtered output
        /// @param in Input vector
        ////////////////////////////////////////////////////////////////////////////////////////////
        Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic> filter(const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> & in)
        {
            uword sz = in.n_elem;
            Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic> out(sz,1);
            for( uword n=0;n<sz;n++)
                out[n] = this->operator()(in[n]);
            return out;
        }
	Eigen::Vector<T3, Eigen::Dynamic> filter(const Eigen::Vector<T1, Eigen::Dynamic> & in)
        {
           Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> in_col = arma::conv_to<Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> >::from(in);
           return arma::conv_to<Eigen::Vector<T3, Eigen::Dynamic> >::from(filter(in_col));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief LMS Filter function setup.
        /// @param _N  Number of filter taps
        /// @param _mu Step size
        /// @param _L Block length
        ////////////////////////////////////////////////////////////////////////////////////////////
        void setup_lms(const uword _N, const double _mu, const  uword _L=1)
        {
            M  = _N;
            mu = _mu;
            L  = _L;
            buf.resize(M,1);buf.zeros();
            b.resize(M,1);b.zeros();
            K.resize(M,1);K.zeros();
            cur_p = 0;
            blk_ctr = 0;
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief LMS Filter update function
        ///
        ///  The LMS filter is updated as <br>
        ///  \f$ \mathbf{b(n)} = \mathbf{b(n-1)}+2\mu\mathbf{x(n)}err(n) \f$ <br>
        ///  where <br> \f$ err(n) = d(n)-\mathbf{b(n-1)^Tx(n)} \f$
        /// @param _err  Feedback error
        ////////////////////////////////////////////////////////////////////////////////////////////
        void lms_adapt(const T3 _err)
        {
            if(do_adapt)
            {
                // Reshape buf
                Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf_tmp(M,1);
                for(uword m=0; m<M; m++)
                {
                    buf_tmp(m) = buf((cur_p+m+1)%M);
                }

                // Accumulate
                K += _err*buf_tmp;

                // Block update
                if(blk_ctr++%L==0)
                {
                      b+=2*mu*K/double(L);
                      K.zeros();
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief NLMS Filter function setup.
        /// @param _N  Number of filter taps
        /// @param _mu Step size
        /// @param _c Regularization factor
        /// @param _L Block length
        ////////////////////////////////////////////////////////////////////////////////////////////
        void setup_nlms(const  uword _N, const double _mu, const T2 _c, const  uword _L=1)
        {
            M  = _N;
            mu = _mu;
            L  = _L;
            c  = _c;
            buf.resize(M,1);buf.zeros();
            b.resize(M,1);b.zeros();
            K.resize(M,1);K.zeros();
            cur_p = 0;
            blk_ctr = 0;
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief NLMS Filter update function
        ///
        ///  The NLMS filter is updated as <br>
        ///  \f$ \mathbf{b(n)} = \mathbf{b(n-1)}+2\mu\frac{\mathbf{x(n)}err(n)}{c+\mathbf{x(n)^Tx(n)}} \f$ <br>
        ///  where <br> \f$ err(n) = d(n)-\mathbf{b(n-1)^Tx(n)} \f$
        /// @param _err  Feedback error
        ////////////////////////////////////////////////////////////////////////////////////////////
        void nlms_adapt(const T3 _err)
        {
            if(do_adapt)
            {
                // Reshape buf
                Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf_tmp(M,1);
                for(uword m=0; m<M; m++)
                {
                    buf_tmp(m) = buf((cur_p+m+1)%M);
                }

                // Accumulate
                T1 S = c + arma::as_scalar(buf_tmp.t()*buf_tmp);
                K += _err*buf_tmp/S;

                // Block update
                if(blk_ctr++%L==0)
                {
                      b+=2*mu*K/double(L);
                      K.zeros();
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief LMS-Newton Filter function setup. (Affine Projection Algorithm)
        /// @param _N  Number of filter taps
        /// @param _mu Step size
        /// @param _c Regularization factor
        /// @param _L Block length
        ////////////////////////////////////////////////////////////////////////////////////////////
        void setup_newt(const  uword _N, const double _mu, const T2 _c, const  uword _L=1)
        {
            M  = _N;
            mu = _mu;
            L  = _L;
            c  = _c;
            buf.resize(M,1);buf.zeros();
            b.resize(M,1);b.zeros();
            K.resize(M,1);K.zeros();
            X_tpz.resize(M,L);X_tpz.zeros();
            cur_p = 0;
            blk_ctr = 0;
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief LMS-Newton Filter update function
        ///
        ///  The LMS-Newton filter is updated as <br>
        ///  \f$ \mathbf{b(n)} = \mathbf{b(n-1)}+2\mu\frac{\mathbf{x(n)}err(n)}{c+\mathbf{R_{xx}}} \f$ <br>
        ///  where <br> \f$ err(n) = d(n)-\mathbf{b(n-1)^Tx(n)} \f$ <br>
        ///  and \f$ \mathbf{R_{xx}} \f$ is the correlation matrix
        /// @param _err  Feedback error
        ////////////////////////////////////////////////////////////////////////////////////////////
        void newt_adapt(const T3 _err)
        {
            if(do_adapt)
            {
                // Reshape buf
                Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf_tmp(M,1);
                for(uword m=0; m<M; m++)
                {
                    buf_tmp(m) = buf((cur_p+m+1)%M);
                }

                // Accumulate in buf
                K += _err*buf_tmp;
                X_tpz.col(blk_ctr%L) = buf_tmp;

                // Block update
                if(blk_ctr++%L==0)
                {
                      // Correlation matrix estimate
                      Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> Rxx = X_tpz*X_tpz.t()/L;
                      Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> I; I.eye(M,M);
                      b+=mu*pinv(Rxx+c*I)*K/L;
                      K.zeros();
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief RLS Filter function setup.
        /// @param _N  Number of filter taps
        /// @param _lmd Lambda
        /// @param _P0 Inverse corr matrix initializer
        ////////////////////////////////////////////////////////////////////////////////////////////
        void setup_rls(const uword _N, const double _lmd,const double _P0)
        {
            M  = _N;
            lmd  = _lmd;
            L = 1;
            P.eye(M,M);
            P =_P0*P;
            K.resize(M,1);K.zeros();
            buf.resize(M,1);buf.zeros();
            b.resize(M,1);b.zeros();
            cur_p = 0;
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief RLS Filter update function
        ///
        ///  The RLS filter is updated as <br>
        ///  \f$ \mathbf{b(n)} = \mathbf{b(n-1)}+\mathbf{Kx(n)}\f$ <br>
        ///  where <br>\f$ \mathbf{K} =\frac{\mathbf{Px}}{\lambda+\mathbf{x^TPx}} \f$ <br>
        ///  and <br>\f$ \mathbf{P^+} =\frac{\mathbf{P^-+xP^-x^T }}{\lambda} \f$ <br>
        /// @param _err  Feedback error
        ////////////////////////////////////////////////////////////////////////////////////////////
        void rls_adapt(const T3 _err)
        {
            if(do_adapt)
            {
                // Reshape buf
                Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf_tmp(M,1);
                for(uword m=0; m<M; m++)
                {
                    buf_tmp(m) = buf((cur_p+m+1)%M);
                }

                // Update P
                T1 S = lmd + arma::as_scalar(buf_tmp.t()*P*buf_tmp);
                K = P*buf_tmp/S;
                P = (P-K*buf_tmp.t()*P)/lmd;

                // Update coeffs
                b = b + K*_err;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Kalman Filter function setup.
        /// @param _N  Number of filter taps
        /// @param _P0 Inverse corr matrix initializer
        /// @param _Q0 Process noise matrix initializer
        /// @param _R0 Measurement noise matrix initializer
        ////////////////////////////////////////////////////////////////////////////////////////////
        void setup_kalman(const uword _N, const double _P0, const double _Q0, const double _R0)
        {
            M  = _N;
            L = 1;
            P.eye(M,M);
            P =_P0*P;
            Q.eye(M,M);
            Q =_Q0*Q;
            R.ones(1,1);
            R =_R0*R;
            K.resize(M,1);K.zeros();
            buf.resize(M,1);buf.zeros();
            b.resize(M,1);b.zeros();
            cur_p = 0;
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Kalman Filter update function
        ///
        ///  The Kalman filter is updated as <br>
        ///  \f$ \mathbf{b(n)} = \mathbf{b(n-1)}+\mathbf{Kx(n)}\f$ <br>
        ///  where <br>\f$ \mathbf{K} =\frac{\mathbf{Px}}{R+\mathbf{x^TPx}} \f$ <br>
        ///  and <br>\f$ \mathbf{P^+} =\mathbf{P^-+xP^-x^T }+Q \f$ <br>
        /// @param _err  Feedback error
        ////////////////////////////////////////////////////////////////////////////////////////////
        void kalman_adapt(const T3 _err)
        {
            if(do_adapt)
            {
                // Reshape buf
                Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> buf_tmp(M,1);
                for(uword m=0; m<M; m++)
                {
                    buf_tmp(m) = buf((cur_p+m+1)%M);
                }

                // Innovation/error covariance
                T1 S = arma::as_scalar(R+buf_tmp.t()*P*buf_tmp);

                // Kalman gain
                K = P*buf_tmp/S;

                // Update coeffs/state
                b = b + K*_err;

                // Update estimate covariance
                P = P-K*buf_tmp.t()*P+Q;
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Get step size
        /// @return Step size mu
        ////////////////////////////////////////////////////////////////////////////////////////////
        double get_step_size(void)
        {
            return mu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Get P
        /// @return P
        ////////////////////////////////////////////////////////////////////////////////////////////
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> get_P(void)
        {
            return P;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Get K
        /// @return K
        ////////////////////////////////////////////////////////////////////////////////////////////
        Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> get_K(void)
        {
            return K;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Set step size
        /// @param _mu Step size mu
        ////////////////////////////////////////////////////////////////////////////////////////////
        void set_step_size(const double _mu)
        {
            mu = _mu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Start adapt
        ////////////////////////////////////////////////////////////////////////////////////////////
        void adapt_enable(void)
        {
            do_adapt = 1;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Stop adapt
        ////////////////////////////////////////////////////////////////////////////////////////////
        void adapt_disble(void)
        {
            do_adapt = 0;
        }

    };


    ///
    /// \brief IIR/ARMA filter class.
    ///
    /// Implements IIR/ARMA filter functions as \f[  a_0y(n) = b_0x(n)+b_1x(n-1)+...+b_{M-1}x(n-(M-1))-a_1y(n-1)-...-a_{M-1}y(n-(M-1))\f]
    /// where M is the number of taps in the FIR filter part and M is the number of taps in the IIR filter. The filter order is (M-1,M-1)
    ///
    template <class T1, class T2, class T3>
    class IIR_filt
    {
    private:
        uword M;                ///< Nr of MA filter taps
        uword N;                ///< Nr of AR filter taps
        uword b_cur_p;          ///< Pointer to current sample in MA buffer
        uword a_cur_p;          ///< Pointer to current sample in AR buffer
	Eigen::Vector<T2, Eigen::Dynamic> b;      ///< MA Filter coefficients
        Eigen::Vector<T2, Eigen::Dynamic> a;      ///< AR Filter coefficients
        Eigen::Vector<T1, Eigen::Dynamic> b_buf;  ///< MA Signal buffer
        Eigen::Vector<T1, Eigen::Dynamic> a_buf;  ///< AR Signal buffer
    public:
        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Constructor.
        ////////////////////////////////////////////////////////////////////////////////////////////
        IIR_filt(){}

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Destructor.
        ////////////////////////////////////////////////////////////////////////////////////////////
        ~IIR_filt(){}

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Clears the internal states and pointers.
        ////////////////////////////////////////////////////////////////////////////////////////////
        void clear(void)
        {
            b_buf = Eigen::Vector<T1, Eigen::Dynamic>::Zero(b_buf.size());
            a_buf = Eigen::Vector<T1, Eigen::Dynamic>::Zero(a_buf.size());
            b_cur_p = 0;
            a_cur_p = 0;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Sets coefficients in IIR filter.
        /// The internal state and pointers are cleared
        /// @param _b Filter coefficients \f$ [b_0 ..b_M] \f$
        /// @param _a Filter coefficients \f$ [a_0 ..a_N] \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
        void set_coeffs(const Eigen::Vector<T2, Eigen::Dynamic> &_b,const Eigen::Vector<T2, Eigen::Dynamic> &_a)
        {
            M = _b.size();
            N = _a.size();
            b_buf.resize(M);
            a_buf.resize(N);
            this->clear();
            b = _b/_a[0];
            a = _a/_a[0];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Updates coefficients in filter without clearing the internal states.
        /// @param _b Filter coefficients \f$ [b_0 ..b_M] \f$
        /// @param _a Filter coefficients \f$ [a_0 ..a_N] \f$
        ////////////////////////////////////////////////////////////////////////////////////////////
        void update_coeffs(const Eigen::Vector<T2, Eigen::Dynamic> &_b,const Eigen::Vector<T2, Eigen::Dynamic> &_a)
        {
            b = _b/_a[0];
            a = _a/_a[0];
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Filter operator.
        /// @return Filtered output
        /// @param in Input sample
        ////////////////////////////////////////////////////////////////////////////////////////////
        T3 operator()(const T1 & in)
        {
            T3 out=0;
            uword p = 0;

            // MA part
            b_buf[b_cur_p] = in;                // Insert new sample
            for(uword m = b_cur_p; m < M; m++)
                out += b[p++]*b_buf[m];         // Calc upper part
            for(uword m = 0; m < b_cur_p; m++)
                out += b[p++]*b_buf[m];         // ... and lower

            // Move insertion point
            if(b_cur_p == 0)
                b_cur_p = M-1;
            else
                b_cur_p--;

            // AR part
            p=1;
            for(uword n = a_cur_p+1; n < N; n++)
                out -= a[p++]*a_buf[n];         // Calc upper part
            for(uword n = 0; n < a_cur_p; n++)
                out -= a[p++]*a_buf[n];         // ... and lower

            a_buf[a_cur_p] = out;		        // Insert output

            // Move insertion point
            if(a_cur_p == 0)
                a_cur_p = N-1;
            else
                a_cur_p--;

            return out;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////
        /// \brief Filter function.
        /// @return Filtered output
        /// @param in Input vector
        ////////////////////////////////////////////////////////////////////////////////////////////
        Eigen::Vector<T3, Eigen::Dynamic> filter(const Eigen::Vector<T1, Eigen::Dynamic> & in)
        {
            uword sz = in.size();
            Eigen::Vector<T3, Eigen::Dynamic> out(sz);
            for( uword n=0;n<sz;n++)
                out[n] = this->operator()(in[n]);
            return out;
        }
    };


    ///
    /// Filter design functions
    ///

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief FIR lowpass design function.
    /// FIR lowpassdesign using windows method (hamming window).
    /// NB! Returns size M+1
    /// @return b Filter coefficients \f$ [b_0 ..b_N] \f$
    /// @param M Filter order
    /// @param f0 Filter cutoff frequency in interval [0..1]
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd fir1(const uword M, const double f0)
    {
        Eigen::VectorXd b(M+1), h(M+1);
        h = hamming(M+1);
        for (uword m=0;m<M+1;m++)
        {
            b[m] = f0*h[m]*sinc(f0*(m-M/2.0));
        }
        return b/arma::sum(b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief FIR highpass design function.
    /// FIR design using windows method (hamming window).
    /// NB! Returns size M+1
    /// @return b Filter coefficients \f$ [b_0 ..b_N] \f$
    /// @param M Filter order (must be even)
    /// @param f0 Filter cutoff frequency in interval [0..1]
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd fir1_hp(const uword M, const double f0)
    {
        if(M%2 != 0)
            err_handler("Filter order must be even");
        Eigen::VectorXd b(M+1), h(M+1);
        h = hamming(M+1);
        for (uword m=0;m<M+1;m++)
        {
            b[m] = h[m]*(sinc(m-M/2.0)-f0*sinc(f0*(m-M/2.0)));
        }

        // Scale
        std::complex<double> i(0,1);
        double nrm;
        Eigen::VectorXd fv=arma::regspace(0,double(M));
        nrm = abs(arma::sum(exp(-i*fv*PI)%b));

        return b/nrm;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief FIR bandpass design function.
    /// FIR design using windows method (hamming window).
    /// NB! Returns size M+1
    /// @return b Filter coefficients \f$ [b_0 ..b_N] \f$
    /// @param M Filter order
    /// @param f0 Filter low cutoff frequency in interval [0..1]
    /// @param f1 Filter high cutoff frequency in interval [0..1]
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd fir1_bp(const uword M, const double f0, const double f1)
    {
        if(f1<=f0)
            err_handler("Frequencies must be [0 < f0 < f1 < 1]");

        Eigen::VectorXd b(M+1), h(M+1);
        h = hamming(M+1);
        for (uword m=0;m<M+1;m++)
        {
            b[m] = h[m]*(f1*sinc(f1*(m-M/2.0))-f0*sinc(f0*(m-M/2.0)));
        }

        // Scale
        double fc = (f0+f1)/2;
        std::complex<double> i(0,1);
        double nrm;
        Eigen::VectorXd fv=arma::regspace(0,double(M));
        nrm = abs(arma::sum(exp(-i*fv*PI*fc)%b));

        return b/nrm;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief FIR bandstop design function.
    /// FIR design using windows method (hamming window).
    /// NB! Returns size M+1
    /// @return b Filter coefficients \f$ [b_0 ..b_N] \f$
    /// @param M Filter order (must be even)
    /// @param f0 Filter low cutoff frequency in interval [0..1]
    /// @param f1 Filter high cutoff frequency in interval [0..1]
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd fir1_bs(const uword M, const double f0, const double f1)
    {
        if(M%2 != 0)
            err_handler("Filter order must be even");
        if(f1<=f0)
            err_handler("Frequencies must be [0 < f0 < f1 < 1]");

        Eigen::VectorXd b(M+1), h(M+1);
        h = hamming(M+1);
        for (uword m=0;m<M+1;m++)
        {
            b[m] = h[m]*(sinc(m-M/2.0)-f1*sinc(f1*(m-M/2.0))+f0*sinc(f0*(m-M/2.0)));
        }

        return b/arma::sum(b);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Fractional delay function.
    /// Fractional delay filter design using windowed sinc method.
    /// Actual delay is M/2+fd samples for even nr of taps and
    /// (M-1)/2+fd for odd nr of taps
    /// Best performance if -1 < fd < 1
    /// @param M Filter length
    /// @param fd Fractional delay
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd fd_filter( const uword M, double fd )
    {
        Eigen::VectorXd h(M);
        Eigen::VectorXd w = blackmanharris(M);
        if( M % 2 == 1 ) fd = fd-0.5; // Offset for odd nr of taps
        for(uword m=0;m<M;m++)
        {
            h(m) = w(m)*sinc(m-M/2.0-fd);
        }
        h = h/arma::sum(h);  // Normalize gain

        return h;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Frequency response function.
    /// Calculates the frequency response
    /// @param b FIR/MA filter coefficients
    /// @param a IIR/AR filter coefficients
    /// @param K Number of evaluation points, Default 512
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXcd freq( const Eigen::VectorXd b, const Eigen::VectorXd a, const uword K=512)
    {
        Eigen::VectorXcd h(K);
        uword M = b.size();
        uword N = a.size();
        std::complex<double> b_tmp,a_tmp,i(0,1);
        for(uword k=0;k<K;k++)
        {
            b_tmp=std::complex<double>(b(0),0);
            for(uword m=1;m<M;m++)
                b_tmp+= b(m)*(cos(m*PI*k/K)-i*sin(m*PI*k/K));
            a_tmp=std::complex<double>(a(0),0);
            for(uword n=1;n<N;n++)
                a_tmp+= a(n)*(cos(n*PI*k/K)-i*sin(n*PI*k/K));
            h(k) = b_tmp/a_tmp;
        }
        return h;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Frequency magnitude response function.
    /// Calculates the frequency magnitude response
    /// @param b FIR/MA filter coefficients
    /// @param a IIR/AR filter coefficients
    /// @param K Number of evaluation points, Default 512
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd freqz( const Eigen::VectorXd b, const Eigen::VectorXd a, const uword K=512)
    {
        Eigen::VectorXcd f = freq(b,a,K);
        return abs(f);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    /// \brief Frequency phase response function.
    /// Calculates the frequency phase response
    /// @param b FIR/MA filter coefficients
    /// @param a IIR/AR filter coefficients
    /// @param K Number of evaluation points, Default 512
    ////////////////////////////////////////////////////////////////////////////////////////////
    EIGEN_STRONG_INLINE Eigen::VectorXd phasez( const Eigen::VectorXd b, const Eigen::VectorXd a, const uword K=512)
    {
        Eigen::VectorXcd f = freq(b,a,K);
        return angle(f);
    }
    /// @}

} // end namespace
#endif

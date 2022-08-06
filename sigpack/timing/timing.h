// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SP_TIMING_H
#define SP_TIMING_H
namespace sp
{
    ///
    /// @defgroup timing Timing
    /// \brief Timing functions.
    /// @{

    ///
    /// \brief A delay class.
    ///
    /// Implements different timing related functions such as delay
    ///
    template <class T1>
    class Delay
    {
        private:
            uword D;        ///< The delay value
            uword cur_p;    ///< Pointer to current sample in buffer
            Eigen::Vector<T1, Eigen::Dynamic> buf;    ///< Signal buffer
        public:
            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Constructor.
            ////////////////////////////////////////////////////////////////////////////////////////////
            Delay()
            {
                cur_p = 0;
                D = 0;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Constructor with delay input.
            /// @param _D delay
            ////////////////////////////////////////////////////////////////////////////////////////////
            Delay(const uword _D)
            {
                set_delay(_D);
                clear();
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Destructor.
            ////////////////////////////////////////////////////////////////////////////////////////////
            ~Delay() {}

            ////////////////////////////////////////////////////////////////////////////////////////////
            ///  \brief Clears internal state.
            ////////////////////////////////////////////////////////////////////////////////////////////
            void clear(void)
            {
                buf = Eigen::Vector<T1, Eigen::Dynamic>::Zero(buf.size());
                cur_p = 0;
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief Sets delay.
            /// @param _D delay
            ////////////////////////////////////////////////////////////////////////////////////////////
            void set_delay(const uword _D)
            {
                D = _D+1;
                buf.resize(D);
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief A delay operator.
            /// @param in sample input
            ////////////////////////////////////////////////////////////////////////////////////////////
            T1 operator()(const T1& in)
            {
                buf[cur_p] = in;                    // Insert new sample
                // Move insertion point
                if (cur_p == 0)
                    cur_p = D-1;
                else
                    cur_p--;

                return buf[cur_p];
            }

            ////////////////////////////////////////////////////////////////////////////////////////////
            /// \brief A delay operator (vector version).
            /// @param in vector input
            ////////////////////////////////////////////////////////////////////////////////////////////
            Eigen::Vector<T1, Eigen::Dynamic> delay(const Eigen::Vector<T1, Eigen::Dynamic>& in)
            {
                uword sz = in.size();
                Eigen::Vector<T1, Eigen::Dynamic> out(sz);
                for(uword n=0; n<sz; n++)
                    out[n] = this->operator()(in[n]);
                return out;
            }
    };
    /// @}

} // end namespace
#endif

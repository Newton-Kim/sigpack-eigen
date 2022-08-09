// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SP_TYPES_H
#define SP_TYPES_H

#include <iostream>
using std::cout;

namespace sp
{
#ifdef __x86_64__
    typedef uint64_t uword;
    typedef int64_t sword;
#else //__x86_64__
    typedef uint32_t uword;
    typedef int32_t sword;
#endif //__x86_64__
}
#endif //SP_TYPES_H


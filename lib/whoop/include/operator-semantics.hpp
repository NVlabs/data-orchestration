/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef WHOOP_OPERATOR_SEMANTICS_HPP_
#define WHOOP_OPERATOR_SEMANTICS_HPP_

#include "typedefs.hpp"

namespace whoop
{

  DataType_t PlusOp(const DataType_t& x, const DataType_t& y);
  DataType_t MinusOp(const DataType_t& x, const DataType_t& y);
  DataType_t MulOp(const DataType_t& x, const DataType_t& y);
  DataType_t DivOp(const DataType_t& x, const DataType_t& y);
  DataType_t ModOp(const DataType_t& x, const DataType_t& y);
  DataType_t EQOp(const DataType_t&x , const DataType_t& y);
  DataType_t IntEQOp(const DataType_t&x , const DataType_t& y);
  DataType_t NEQOp(const DataType_t&x , const DataType_t& y);
  DataType_t GTEOp(const DataType_t& x, const DataType_t& y);
  DataType_t LTEOp(const DataType_t&x , const DataType_t& y);
  DataType_t GTOp(const DataType_t& x, const DataType_t& y);
  DataType_t LTOp(const DataType_t& x, const DataType_t& y);
  DataType_t ANDOp(const DataType_t& x, const DataType_t& y);
  DataType_t OROp(const DataType_t& x, const DataType_t& y);
  DataType_t BWANDOp(const DataType_t& x, const DataType_t& y);
  DataType_t BWOROp(const DataType_t& x, const DataType_t& y);
  DataType_t NOPOp(const DataType_t& x, const DataType_t& y);
  DataType_t POSTINCOp(const DataType_t& x);
  DataType_t PREINCOp(const DataType_t& x);
};

#endif

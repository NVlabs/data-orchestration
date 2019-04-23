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

namespace whoop
{

  int PlusOp(const int& x, const int& y);
  int MinusOp(const int& x, const int& y);
  int MulOp(const int& x, const int& y);
  int DivOp(const int& x, const int& y);
  int ModOp(const int& x, const int& y);
  int EQOp(const int&x , const int& y);
  int NEQOp(const int&x , const int& y);
  int GTEOp(const int& x, const int& y);
  int LTEOp(const int&x , const int& y);
  int GTOp(const int& x, const int& y);
  int LTOp(const int& x, const int& y);
  int ANDOp(const int& x, const int& y);
  int OROp(const int& x, const int& y);
  int BWANDOp(const int& x, const int& y);
  int BWOROp(const int& x, const int& y);
  int NOPOp(const int& x, const int& y);
  int POSTINCOp(const int& x);
  int PREINCOp(const int& x);
};

#endif

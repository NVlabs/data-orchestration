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

#ifndef TIMEWHOOP_NETWORK_ON_CHIP_SPEC_
#define TIMEWHOOP_NETWORK_ON_CHIP_SPEC_



namespace timewhoop
{

/*** Network-on-chip spec
- Description: This file defines some fundamental parameters for
  the network-on-chip model in timewhoop.

*/


  /* zero-load latencies */
  const int zeroload_bus = 3; // Egress NIC (bus arbitration) + Traversal + Ingress NIC = 3 cycle
  const int zeroload_crossbar = 3; // Input NIC (output arbitration) + Traversal + Output NIC = 3 cycle

  const double wire_energy_per_nm = 0.001; //pJ; currently it is a dummy number // TODO: Add real number

  const double multiplier_avg_wire_length_bus = 1.23; //nm; currently a dummy number // TODO: Add real number
  const double multiplier_avg_wire_length_crossbar = 1.23; //nm; currently a dummy number // TODO: Add real number
  const double multiplier_avg_wire_length_mesh = 0.57; //nm; currently a dummy number // TODO: Add real number

};


#endif

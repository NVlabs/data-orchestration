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

#ifndef TIMEWHOOP_NETWORK_ON_CHIP_
#define TIMEWHOOP_NETWORK_ON_CHIP_

#include <cmath>

#include "network-on-chip-spec.hpp"


/*** Nework-on-Chip 
- Description: This file includes network-on-chip model for 
  runtime(latency) analysis

- This file is for future extension of timewhoop

*/

namespace timewhoop
{

  class NetworkOnChipModel
  {
    protected:
      int bandwidth_ = 1; // Multiple of the size of each data element
      int average_latency_ = 2; // Average zero-load latency
      bool multicast_support_ = false; // Multicast capability

      double average_traverse_wire_length = 0.001; //nm; currently it is a dummy number // TODO: Add real number

    public:
      NetworkOnChip(int& bw, int& avg_latency, bool& mc_support) :
        bandwidth_(bw), average_latency_(avg_latency), multicast_support_(mc_support)
      {
      }

      int GetBandwidth()
      {
        return bandwidth_;
      }

      void GetLatency(int send_volume, int multicast_factor)
      {
        //Consider multicasting        
        int actual_sends;
        if(multicast_support_)
        {
          actual_sends = static_cast<int>( static_cast<double> (send_volume) / static_cast<double> (multicast_factor));
        }
        else
        {
          actual_sends = send_volume;
        }

        int num_foldings = actual_sends/bandwidth_;
        int initial_latency = average_latency_;
        
        //Consider pipelining aspects in NoC
        int latency = initial_latency + num_foldings;

        return latency;
      }
  };

  class BusModel : public NetworkOnChipModel
  {
    protected:
    

    public:
      BusModel(int& bw) :
        NetworkOnChipModel(bw, zeroload_bus, true)
      {
      }
  };

  class CrossbarModel : public NetworkOnChipModel
  {
    protected:


    public:
      CrossbarModel(int& num_connected) :
        NetworkOnChipModel(num_connected, zeroload_crossbar, false);
      {
      }
  };

  class MeshModel : public NetworkOnChipModel
  {
    private:
      int GetBisectionBandwidth(int& num_mesh_nodes)
      {
        double ret = std:sqrt(static_cast<double>(num_mesh_nodes));
        return static_cast<int>(ret);
      }
    protected:

    public:
      MeshModel(int& num_conntected) :
        NetworkOnChipModel(GetBisectionBandwidth(num_conntected))
      {
      }
  };

};

#endif

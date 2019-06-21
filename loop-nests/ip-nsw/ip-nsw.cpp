/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *	* Redistributions of source code must retain the above copyright
 *	notice, this list of conditions and the following disclaimer.
 *	* Redistributions in binary form must reproduce the above copyright
 *	notice, this list of conditions and the following disclaimer in the
 *	documentation and/or other materials provided with the distribution.
 *	* Neither the name of NVIDIA CORPORATION nor the names of its
 *	contributors may be used to endorse or promote products derived
 *	from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.	IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "whoop.hpp"
#include <math.h>

int GetMin( int a, int b) {
	if( a < b ) return a;
	return b;
}

int main(int argc, char** argv) {

	using namespace whoop;
	VecIn	candidates("candidates");
	VecIn	offsets("offsets");
	VecIn	Coordinates("coordinates");
	VecIn	data("data");
	VecOut dataOut("dataout");
	Vec	visited("visited");

	whoop::Init(argc, argv);

	int numcandidates = candidates.Size();
	int numVertices = offsets.Size() - 1;
	int numEdges = Coordinates.Size();

	// Short-form variable names
	const int C = numcandidates;
	const int V = numVertices;
	const int D = 2; // data dimension 


	whoop::T(0) << "Number of candidates: " << C << whoop::EndT;
	whoop::T(0) << "Number of Vertices: " << V << whoop::EndT;
	whoop::T(0) << "Number of Edges:	" << numEdges << whoop::EndT;
	whoop::T(0) << whoop::EndT;
	
	// Initialize visited vector
	visited.Resize( V );
	dataOut.Resize( D );

	for(int v=0; v < V; v++) {
		visited.At(v) = 0;
	}

	whoop::T(0) << "RUNNING..." << whoop::EndT;

	const int src = 0;

	Var j("j");
	Var s("s");
	Var i("i");
	Var i0("i0");
	Var i1("i1");
	Var c("c");
	Var v("v");
	Var d("d");

	t_for(j, 0, C); {
		s = candidates[j];
		i0 = offsets[s];
		i1 = offsets[s+1];

		t_for(i, i0, i1); {
		  c = Coordinates[i];
			v = visited[c];

			w_if(v == 0); {
				visited[c] = 1;

				t_for (d, 0, D); { 
					dataOut[d] = data[D*c + d] + (dataOut[d] * 0);
				} end();

			} end();

		} end();

	} end();

	whoop::T(0) << "RUNNING..." << whoop::EndT;
	whoop::Run();
	whoop::T(0) << "DONE." << whoop::EndT;

	//for (int v = 0; v < V; v++)
	//{
	//	whoop::T(3) << "connected " << v << " = " << " Parent: " << whoop::EndT;
	//}
	
	whoop::Done();
}


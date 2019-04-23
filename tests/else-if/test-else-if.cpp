/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
 #include "whoop.hpp"

enum
{
  Failed,
  Passed
} TestResult;

int main(int argc, char** argv)
{

  using namespace whoop;
  
  whoop::Init(argc, argv);
  
  Var i("i");
  Var x("x");
  VecOut res("res");
  res.Resize(6);
  
  t_for(i, 0, 1); // For-loop assignment
  {
    x = 0;
     
    // Test 0: basic if
    w_if(x == 0);
    {
      res[0] = Passed;
    }
    end();
    
    // Test 1: if-else
    w_if(x == 1);
    {
      res[1] = Failed;
    }
    w_else();
    {
      res[1] = Passed;
    }
    end();
    
    // Test 2: if-else-if
    w_if(x == 1);
    {
      res[2] = Failed;
    }
    w_else_if(x == 0);
    {
      res[2] = Passed;
    }
    end();
    
    // Test 3: if-else-if
    w_if(x == 1);
    {
      res[3] = Failed;
    }
    w_else_if(x == 0);
    {
      res[3] = Passed;
    }
    end();
    
    // Test 4: if-else-if-else-if
    w_if(x == 1);
    {
      res[4] = Failed;
    }
    w_else_if(x == 2);
    {
      res[4] = Failed;
    }
    w_else_if(x == 0);
    {
      res[4] = Passed;
    }
    end();
    
    // Test 5: if-else-if-else-if-else
    w_if(x == 1);
    {
      res[5] = Failed;
    }
    w_else_if(x == 2);
    {
      res[5] = Failed;
    }
    w_else_if(x == 3);
    {
      res[5] = Failed;
    }
    w_else();
    {
      res[5] = Passed;
    }
    end();
    
  }
  end();

  std::cout << "RUNNING..." << std::endl;
  whoop::Run();
  std::cout << "DONE." << std::endl;
  
  whoop::Done();
}

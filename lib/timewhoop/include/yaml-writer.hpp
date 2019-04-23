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

#ifndef TIMEWHOOP_YAML_WRITER_HPP_
#define TIMEWHOOP_YAML_WRITER_HPP_

#include <string>
#include <iostream>
#include <fstream>

/**** YAML writer
- Description: This file includes a YamlWriter class, which is a
  helper class to write YAML files

*/

namespace timewhoop
{

  class YamlWriter
  {
    private:
      std::ofstream output_file;
      int current_indent_level;

      std::string indent(int indent_level)
      {
        std::string retStr;
        for(int indent = 0; indent < indent_level; indent++)
        {
          retStr = retStr + "\t";
        }

        return retStr;
      }

    public:
      YamlWriter(const std::string& filename = "output.yaml")
      {
        current_indent_level = 0;
        output_file.open(filename);
        if(output_file.fail())
        {
          //TODO: define the behavior for fail
          exit(-1);
        }
      }

      // Add a plain string to the output flie.
      //   level: Number of indentations before the added string
      //   str: The target string to be written
      void write_mapping_contents(int level, const std::string& str)
      {
        auto wrStr = indent(level);
        wrStr += str;
        output_file << wrStr;
      }

      // Write a key for a mapping
      //  level: Number of indentations before the added key
      //  str: The target string to be written
      //  new_line: Indicates if the writer needs to change the line after adding a key
      void write_mapping_key(int level, const std::string& str, bool new_line=true)
      {
        auto wrStr = indent(level);

        wrStr = wrStr + str + ": ";
        if(new_line)
          wrStr += "\n";

        output_file << wrStr;
      }


      // Write a key for a list item
      //  level: Number of indentations before the added item
      //  str: The target string to be written
      //  is_sublist_item: Indicates if the str is concatenated as a sub-item in a list or the first item of a list
      void write_list_item(int level, const std::string& str, bool is_sublist_item)
      {
        std::string wrStr;

        if(is_sublist_item)
        {
          wrStr = " "+ str;
        }
        else
        {
          wrStr = indent(level);
          wrStr = wrStr + "- " + str + "\n";
        }

        output_file << wrStr;
      }

      // Write a key for a list item
      //  level: Number of indentations before the added item
      //  comments: The comment to be written
      void write_comments(int level, const std::string& comment)
      {
        auto wrStr = indent(level);

        wrStr = wrStr + "#" + comment + "\n";
 
        output_file << wrStr;
      }

      // Write a new line character (change the line)
      void write_blank_line()
      {
        output_file << "\n"; 
      }
  };
};
#endif /* TIMEWHOOP_YAML_WRITER_HPP_ */

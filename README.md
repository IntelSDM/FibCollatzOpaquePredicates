# Fibonacci Collatz
Built for LLVM 19.1 - Windows x64 <br> <br>

We combine recursive fibonacci to create our dynamic/runtime value, then collatz to determine the characteristics of the fibinaci sequence value, determining if its odd, even. <br>
This works as symbolically executing the fibonacci sequence to determine the runtime would take hours. <br>
This paper: [Advancing Code Obfuscation: Novel Opaque Predicate Techniques to Counter Dynamic Symbolic Execution](https://www.sciopen.com/local/article_pdf/10.32604/cmc.2025.062743.pdf) Found after 3 hours, angr and klee would simply timeout, the same effect would happen on any form of AI; it's a genuine unsolvable mathematical problem simplify. <br>
<br>
<img width="903" height="925" alt="image" src="https://github.com/user-attachments/assets/2b5a6050-a79b-4959-ae5e-8b695a4aacec" /><br>
<img width="1126" height="944" alt="image" src="https://github.com/user-attachments/assets/205b7698-5000-4f87-8cc2-11fc880f2627" /><br>
<img width="1137" height="952" alt="image" src="https://github.com/user-attachments/assets/cf4290f3-a180-4a9c-9cc8-aa19d62b4c33" /><br>
<img width="1002" height="975" alt="image" src="https://github.com/user-attachments/assets/ecbdc1e5-24df-4548-900e-549bdcfbac99" /><br>

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll

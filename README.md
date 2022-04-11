# Computing applications with GPU acceleration with Python (Numba-Cuda)

Uses of Numba and Cuda to speed up your code. Built-in GUI for better understanding.

## Description

Python is an extremely known programming language used in many ways (for example Maching Learning).
Despite that it's pretty slow and causes many programmers to flee to other programming languages
which have faster compilation and running time of the code. To deal with this, there were created
some libraries that act as an accelator in order to make the code run faster than before.
Numba library in one of the most known accelators and is usable with just one line of code
before every method that you write. It compiles the code and then runs it. If we are talking about
small methods that are going to run a few times, it will probably make it slower since it adds the compilation time,
but if there is a big amount of code that is going to run repeatedly then it's faster since the compilation time + running time
are going to be a lot faster than before.
We can also use Cuda if we want to run our code using the GPU, which has more threads than the CPU
but is slower. This means that if our code can run in parallel, it can use many threads and fasten the running time.
But if we have commands that need to run serially then the CPU is the better way to run our code.

## Dependency

- python 3.8.10
- numba 0.46.0 (follow this link: https://numba.pydata.org/numba-doc/dev/user/installing.html?highlight=download%20cuda)
- numpy 1.21.5

## Getting Started

The programm runs just by running the code. It takes some time at the beginning since it's calculating
some parameteres that cannot run while the GUI is open. After that, a GUI is going to show on your screen
with different buttons, each one having a different use.

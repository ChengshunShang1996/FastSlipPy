__copyright__= "Copyright (c) 2026-Present, Chengshun Shang"

__license__ = """


Copyright (c) 2026 Chengshun Shang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .fast_slip_py import FastSlipPy

__all__ = ["FastSlipPy"]

__all__         =["pre_processing","post_processing","solver","utilities"]

__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__license__     = "MIT License"
__URL__         = 'https://github.com/ChengshunShang1996/FastSlipPy'
__logo__        = '''
 ______         _    _____ _         _____       
|  ____|       | |  / ____| (_)     |  __ \      
| |__ __ _ ___ | |_| (___ | |_ _ __ | |__) |   _ 
|  __/ _` / __|| __|\___ \| | | '_ \|  ___/ | | |
| | | (_| \__ \| |_ ____) | | | |_) | |   | |_| |
|_|  \__,_|___/ \__|_____/|_|_| .__/|_|    \__, |
                              | |           __/ |
                              |_|          |___/ '''


print(f"\n"
      f"{__logo__}\n"
      f"FastSlipPy\n"
      f"Version: {__version__}\n"
      f"Author: {__author__}\n"
      f"Copyright: {__copyright__}\n"
      f"URL: {__URL__}\n"
      f"----------------------------------------------------------------------------\n")
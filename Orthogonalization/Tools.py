#!/usr/bin/env python2
import math, sys, os, copy, shutil, struct

def DeleteMakeDir(Path):
    if os.path.exists(Path):
        shutil.rmtree(Path)
    os.makedirs(Path)
    return

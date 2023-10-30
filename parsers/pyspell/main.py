#!/usr/bin/env python

import sys
import spell as s

if __name__ == '__main__':
    slm = s.lcsmap('[\\s]+')
    #s.save('test.pickle', slm)
    #slm = s.load('test.pickle')
    for i in sys.stdin.readlines():
        sub = i.strip('\n')
        obj = slm.insert(sub)
        print(obj.get_id(), obj.param(sub))

#print(slm.dump())

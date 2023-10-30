#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import pickle

class lcsobj():

    def __init__(self, objid, seq, lineid, refmt):
        self._refmt = refmt
        if isinstance(seq, str) == True:
            self._lcsseq = re.split(self._refmt, seq.lstrip().rstrip())
        else:
            self._lcsseq = seq
        self._lineids = [lineid]
        self._pos = []
        self._sep = "	"
        self._id = objid
        return

    def getlcs(self, seq):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        count = 0
        lastmatch = -1
        for i in range(len(self._lcsseq)):
            #if self._lcsseq[i] == '*':
            if self._ispos(i) == True:
                continue
            for j in range(lastmatch+1, len(seq)):
                if self._lcsseq[i] == seq[j]:
                    lastmatch = j
                    count += 1
                    break
        return count

    def insert(self, seq, lineid):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        self._lineids.append(lineid)
        temp = ""
        lastmatch = -1
        placeholder = False

        for i in range(len(self._lcsseq)):
            #if self._lcsseq[i] == '*':
            if self._ispos(i) == True:
                if not placeholder:
                    temp = temp + "* "
                placeholder = True
                continue
            for j in range(lastmatch+1, len(seq)):
                if self._lcsseq[i] == seq[j]:
                    placeholder = False
                    temp = temp + self._lcsseq[i] + " "
                    lastmatch = j
                    break
                elif not placeholder:
                    temp = temp + "* "
                    placeholder = True
        temp = temp.lstrip().rstrip()
        self._lcsseq = re.split(" ", temp)

        self._pos = self._get_pos()
        self._sep = self._get_sep()

    def tojson(self):
        temp = ""
        for i in self._lcsseq:
            temp = temp + i + " "
        ret = {}
        ret["lcsseq"] = temp
        ret["lineids"] = self._lineids
        ret["postion"] = self._pos
        return json.dumps(ret)

    def length(self):
        return len(self._lcsseq)

    def param(self, seq):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())

        j = 0
        ret = []
        for i in range(len(self._lcsseq)):
            slot = []
            if self._ispos(i) == True:
                while j < len(seq):
                    if i != len(self._lcsseq)-1 and self._lcsseq[i+1] == seq[j]:
                        break
                    else:
                        slot.append(seq[j])
                    j+=1
                ret.append(slot)
            elif self._lcsseq[i] != seq[j]:
                return None
            else:
                j += 1

        if j != len(seq):
            return None
        else:
            return ret

    def re_param(self, seq):
        if isinstance(seq, list) == True:
            seq = ' '.join(seq)
        seq = seq.lstrip().rstrip()

        ret = []
        print(self._sep)
        print(seq)
        p = re.split(self._sep, seq)
        for i in p:
            if len(i) != 0:
                ret.append(re.split(self._refmt, i.lstrip().rstrip()))
        if len(ret) == len(self._pos):
            return ret
        else:
            return None



    def _ispos(self, idx):
        for i in self._pos:
            if i == idx:
                return True
        return False

    def _tcat(self, seq, s, e):
        sub = ''
        for i in range(s, e + 1):
            sub += seq[i] + " "
        return sub.rstrip()

    def _get_sep(self):
        sep_token = []
        s = 0
        e = 0
        for i in range(len(self._lcsseq)):
            if self._ispos(i) == True:
                if s != e:
                    sep_token.append(self._tcat(self._lcsseq, s, e))
                s = i + 1
                e = i + 1
            else:
                e = i
            if e == len(self._lcsseq) - 1:
                sep_token.append(self._tcat(self._lcsseq, s, e))
                break

        ret = ""
        for i in range(len(sep_token)):
            if i == len(sep_token)-1:
                ret += sep_token[i]
            else:
                ret += sep_token[i] + '|'
        return ret

    def _get_pos(self):
        pos = []
        for i in range(len(self._lcsseq)):
            if self._lcsseq[i] == '*':
                pos.append(i)
        return pos

    def get_id(self):
        return self._id

class lcsmap():

    def __init__(self, refmt):
        self._refmt = refmt
        self._lcsobjs = []
        self._lineid = 0
        self._id = 0
        return

    def insert(self, entry):
        seq = re.split(self._refmt, entry.lstrip().rstrip())
        obj = self.match(seq)
        if obj == None:
            self._lineid += 1
            obj = lcsobj(self._id, seq, self._lineid, self._refmt)
            self._lcsobjs.append(obj)
            self._id += 1
        else:
            self._lineid += 1
            obj.insert(seq, self._lineid)

        return obj

    def match(self, seq):
        if isinstance(seq, str) == True:
            seq = re.split(self._refmt, seq.lstrip().rstrip())
        bestmatch = None
        bestmatch_len = 0
        seqlen = len(seq)
        for obj in self._lcsobjs:
            objlen = obj.length()
            if objlen < seqlen/2 or objlen > seqlen*2: continue

            l = obj.getlcs(seq)
            if l >= seqlen/2 and l > bestmatch_len:
                bestmatch = obj
                bestmatch_len = l
        return bestmatch

    def objat(self, idx):
        return self._lcsobjs[idx]

    def size(self):
        return len(self._lcsobjs)

    def dump(self):
        count = 0
        for i in self._lcsobjs:
            print(count, i.tojson())
            count += 1


def save(filename, spell_lcsmap):
    if type(spell_lcsmap) == lcsmap:
        with open(filename,'wb') as f:
            pickle.dump(spell_lcsmap, f)
    else:
        if __debug__ == True:
            print("%s isnt slm object"%filename)

def load(filename):
    with open(filename,'rb') as f:
        slm = pickle.load(f)
        if type(slm) == lcsmap:
            return slm
        else:
            if __debug__ == True:
                print("%s isnt slm object"%filename)
            return None

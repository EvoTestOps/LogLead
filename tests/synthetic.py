"""Deterministic synthetic logs in the Thunderbird, BGL and HDFS formats, for tests.

Besides ordinary lines, each file carries the lines a streaming reader is most likely to get wrong
in a different way from an eager one: non-ASCII text, lines with fewer fields than the format,
quote characters, carriage returns, tabs, trailing spaces, and very long lines. Invalid UTF-8 and
an unbalanced quote (both found in the real Thunderbird log) are opt-in with awkward=True. Content depends
only on the seed and the line count, so a file can be regenerated anywhere.

    uv run tests/synthetic.py thunderbird 1000000 /tmp/tb.log
"""

import os
import random
import sys

_TB_COMPONENTS = ["kernel:", "sshd[1234]:", "crond[99]:", "ib_sm.x[24904]:", "pbs_mom:", "postfix/smtpd[77]:"]
_MESSAGES = [
    "session opened for user root by (uid=0)",
    "[ib_sm_sweep.c:1831]: No topology change",
    "Accepted publickey for root from 10.100.{a}.{b} port {c} ssh2",
    "ACPI: Processor [CPU{a}] (supports 8 throttling states)",
    "RPC: Invalid argument error code {a} at node tbird-sm{b}",
    "instruction cache parity error corrected 0x{c:x}",
    "connection from \"host-{a}\" refused",
    "pam_unix(sshd:session): session closed for user user{b}",
]
_BGL_TYPES = ["RAS", "KERNEL", "APP"]
_BGL_LEVELS = ["INFO", "WARNING", "SEVERE", "FATAL", "ERROR"]
_BGL_COMPONENTS = ["KERNEL", "APP", "MMCS", "LINKCARD", "DISCOVERY"]


def _message(rng):
    return rng.choice(_MESSAGES).format(a=rng.randint(0, 999), b=rng.randint(0, 99), c=rng.randint(0, 99999))


def _edge_case(rng, index, fields, awkward):
    """One line of a kind a streaming reader is likely to get wrong; fields gives a valid prefix."""
    kind = index % 8
    if kind == 0:
        if awkward:
            return fields() + b" bad bytes \xff\xfe (\x1e\x9b\x08 in message"
        return fields() + " non-ascii äö € 漢字 in message".encode()
    if kind == 1:
        return b"- 1131566461 2005.11.09"                       # too few fields
    if kind == 2:
        if awkward:
            return fields() + b' connection from "host-7 unbalanced quote'
        return fields() + b' quoted "value, with comma" and \'single\''
    if kind == 3:
        return fields() + b" carriage return\r"
    if kind == 4:
        return fields() + b" tab\tseparated\tmessage"
    if kind == 5:
        return fields() + b" " + b"long" * 5000
    if kind == 6:
        return fields() + b" trailing spaces   "
    return fields() + b"  double  spaces  "


def _thunderbird_fields(rng, i):
    label = "-" if rng.random() > 0.05 else rng.choice(["VAPI", "R_HDA_NR", "PBS_CON"])
    node = f"dn{i % 731}"
    return (f"{label} {1131566461 + i // 7} 2005.11.{9 + (i // 400000) % 20:02d} {node} Nov 9 12:01:01 "
            f"{node}/{node} {rng.choice(_TB_COMPONENTS)}").encode()


def _bgl_fields(rng, i):
    label = "-" if rng.random() > 0.07 else rng.choice(["KERNDTLB", "APPREAD", "KERNSTOR"])
    node = f"R{rng.randint(0, 77):02d}-M1-N{rng.randint(0, 15)}-C:J{rng.randint(0, 17):02d}-U11"
    ts = 1117838570 + i // 5
    return (f"{label} {ts} 2005.06.03 {node} 2005-06-03-15.42.50.363779 {node} "
            f"{rng.choice(_BGL_TYPES)} {rng.choice(_BGL_COMPONENTS)} {rng.choice(_BGL_LEVELS)}").encode()


def _hdfs_fields(rng, i, block):
    seconds = 20 * 3600 + i // 10 % 14400
    clock = f"{seconds // 3600:02d}{seconds // 60 % 60:02d}{seconds % 60:02d}"
    return (f"0811{9 + (i // 200000) % 2:02d} {clock} {rng.randint(100, 9999)} "
            f"INFO dfs.DataNode$PacketResponder:").encode(), block


def write(kind, lines, path, seed=0, edge_every=997, awkward=False):
    """Create a synthetic log of a given kind ('thunderbird', 'bgl', 'hdfs') and line count.

    For 'hdfs' an anomaly_label.csv is written next to it, since the HDFS loader needs one.
    Returns the path(s) written.
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blocks = [f"blk_{'-' if b % 3 == 0 else ''}{rng.randint(10**17, 10**18)}" for b in range(max(1, lines // 20))]
    with open(path, "wb") as out:
        for i in range(lines):
            if kind == "thunderbird":
                fields = lambda: _thunderbird_fields(rng, i)
            elif kind == "bgl":
                fields = lambda: _bgl_fields(rng, i)
            elif kind == "hdfs":
                block = blocks[rng.randrange(len(blocks))]
                fields = lambda: _hdfs_fields(rng, i, block)[0]
            else:
                raise ValueError(kind)
            if i % edge_every == edge_every - 1 and kind != "hdfs":
                line = _edge_case(rng, i // edge_every, fields, awkward)
            elif kind == "hdfs":
                line = fields() + f" Received block {block} of size {rng.randint(1, 67108864)} from /10.250.{rng.randint(0, 19)}.{rng.randint(0, 255)}".encode()
            else:
                line = fields() + b" " + _message(rng).encode()
            out.write(line + b"\n")
    if kind == "hdfs":
        labels = os.path.join(os.path.dirname(os.path.abspath(path)), "anomaly_label.csv")
        with open(labels, "w") as out:
            out.write("BlockId,Label\n")
            for b, block in enumerate(blocks):
                out.write(f"{block},{'Anomaly' if b % 29 == 0 else 'Normal'}\n")
        return path, labels
    return path


def write_multiline(events, path, seed=0, continuations=True):
    """A log of events with stack-trace lines under them, for RawLoader's line policies.

    The BGL timestamp starts each event. The file opens with continuation lines that have no event
    above them. Some traces are longer than 64 KB, so small sink() chunks end inside an event.
    With continuations=False every line starts an event.
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as out:
        if continuations:
            out.write(b"\tat org.example.Orphan.run(Orphan.java:1)\n  ... 2 more\n")
        for i in range(events):
            micro = i % 1_000_000
            out.write(f"2005-06-03-15.{i // 60000 % 60:02d}.{i // 1000 % 60:02d}.{micro:06d} "
                      f"R{rng.randint(0, 77):02d} {_message(rng)}\n".encode())
            if not continuations:
                continue
            roll = rng.random()
            depth = 3000 if roll < 0.002 else rng.randint(1, 6) if roll < 0.4 else 0
            for d in range(depth):
                out.write(f"\tat org.example.C{d}.m(C{d}.java:{rng.randint(1, 999)})\n".encode())
            if depth and rng.random() < 0.2:
                out.write(b"\n  ... 38 more\n")
    return path


if __name__ == "__main__":
    write(sys.argv[1], int(sys.argv[2]), sys.argv[3])

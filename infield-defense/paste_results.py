#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:30:00 2026

@author: reyer
"""

# Runs one of the analysis scripts and pastes what each step prints back into the script as its #### block.
# The script is cut at its #### blocks (a placeholder block reads "# PENDING"); each piece of code runs in
# turn in one shared namespace, and whatever it prints replaces the block that follows it. The whole
# printout also goes to <script>_run_log.txt. Running it again after a change refreshes every block, so
# no block can hold a number from an older run.
#
#   python paste_results.py infield_expected_outs.py

import contextlib
import io
import os
import re
import sys

script = os.path.abspath(sys.argv[1])
source = open(script).read()
pieces = re.split(r'\n####\n(?:#.*\n)*?####\n', source)
namespace = {'__file__': script, '__name__': '__main__'}
rebuilt, log = [], []
for i, piece in enumerate(pieces):
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(piece, script, 'exec'), namespace)
    text = printed.getvalue().rstrip('\n')
    log.append(text)
    rebuilt.append(piece)
    if i < len(pieces) - 1:
        lines = text.split('\n') if text else ['(nothing printed)']
        rebuilt.append('\n####\n' + '\n'.join(('# ' + line).rstrip() for line in lines) + '\n####\n')
open(script, 'w').write(''.join(rebuilt))
open(script.replace('.py', '_run_log.txt'), 'w').write('\n'.join(t for t in log if t) + '\n')
print(os.path.basename(script), 'blocks pasted', len(pieces) - 1)

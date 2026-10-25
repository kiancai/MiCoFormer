"""Reuse the approved SSH route and authenticated master; never store credentials."""
from pathlib import Path
import base64
import shlex
import subprocess

SOCKET = '/tmp/mcf-resp-methods-20261025.sock'
KNOWN_HOSTS = '/tmp/h100-wyn-01a162de-known-hosts'
OPTIONS = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'ForwardAgent=no', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=15']


def command(remote_command, *, capture=True, timeout=120):
    windows = shlex.join(['ssh', *OPTIONS, 'big2', remote_command])
    tencent = shlex.join(['ssh', *OPTIONS, '-F', '/home/kiancai/.ssh/config', 'desktop_kian', windows])
    args = ['ssh', *OPTIONS, '-F', '/dev/null', '-S', SOCKET, '-o', 'UserKnownHostsFile='+KNOWN_HOSTS,
            'kiancai@82.157.113.42', tencent]
    return subprocess.run(args, text=True, stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None, timeout=timeout, check=True)


def python(source, *, timeout=120):
    encoded = base64.b64encode(source.encode()).decode()
    program = 'import base64;exec(compile(base64.b64decode('+repr(encoded)+'),"mcf_remote","exec"))'
    return command('python3 -c '+shlex.quote(program), timeout=timeout)

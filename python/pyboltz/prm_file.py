import re

def read(file='input.prm'):
    """
    Returns: (dt,T,  beta, K, L)

    Keyword Arguments:
    file -- (default 'input.prm')
    """

    f = open(file, 'r')
    res = {}
    for l in f.readlines():
        m = re.match(r'set dt\s*=(.*)', l)
        if m:
            res['dt'] = eval(m.group(1))
        m = re.match(r'set T\s*=(.*)', l)
        if m:
            res['T'] = eval(m.group(1))
        m = re.match(r'set Tbegin\s*=(.*)', l)
        if m:
            res['Tbegin'] = eval(m.group(1))
        m = re.match(r'set Tfinal\s*=(.*)', l)
        if m:
            res['Tfinal'] = eval(m.group(1))
        m = re.match('set steps\s*=(.*)', l)
        if m:
            res['steps'] = eval(m.group(1))
        m = re.match(r'set nK\s*=(.*)', l)
        if m:
            res['K'] = eval(m.group(1))
        m = re.match(r'set nL\s*=(.*)', l)
        if m:
            res['L'] = eval(m.group(1))
        m = re.match(r'set beta\s*=(.*)', l)
        if m:
            res['beta'] = eval(m.group(1))
        m = re.match(r'set has_scattering\s*=(.*)', l)
        if m:
            if m.group(1) == 'true':
                res['has_scattering'] = True
            else:
                res['has_scattering'] = False
        m = re.match(r'set dump_step\s*=(.*)', l)
        if m:
            res['step'] = eval(m.group(1))

    return res

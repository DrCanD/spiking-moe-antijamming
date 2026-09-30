"""INA260 sampling and clock lookup used by the classifier-tail campaign."""
from pathlib import Path
import os
import time

BOARD=Path(__file__).resolve().parent
SYSTEM=Path(os.sep)/'sys'


def pl_clock_hz():
    for path in [SYSTEM/'kernel/debug/clk/pl0_ref/clk_rate',BOARD/'firmware/pl_clk_actual_hz.txt',BOARD/'firmware/pl_clk_hz.txt']:
        try:return float(path.read_text().strip()),str(path)
        except (OSError,ValueError):pass
    return 83.332e6,'default'


F_CLK,F_CLK_SRC=pl_clock_hz()


def find_hwmon(name='ina260_u14'):
    for directory in (SYSTEM/'class/hwmon').glob('hwmon*'):
        try:
            if (directory/'name').read_text().strip()==name and (directory/'power1_input').exists():
                return str(directory/'power1_input')
        except OSError:pass
    raise FileNotFoundError('INA260 sensor not found')


def set_performance():
    for path in (SYSTEM/'devices/system/cpu').glob('cpu*/cpufreq/scaling_governor'):
        try:path.write_text('performance')
        except OSError:pass


def log_power(path,seconds,hz=10.):
    values=[];start=time.time();k=0
    while True:
        target=start+k/hz;now=time.time()
        if now<target:time.sleep(target-now)
        if time.time()-start>=seconds:break
        values.append(int(Path(path).read_text())/1000.)
        k+=1
    return values

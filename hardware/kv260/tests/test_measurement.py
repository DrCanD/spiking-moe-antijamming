"""Offline tests of measurement acceptance and units. No physical measurements."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'board'))
import run_gate as g

class MockReplay:
    def __init__(self,max_rate=1.5):self.max_rate=max_rate
    def call(self,cmd,arg0,blk_en,pace,n):
        self.frames=arg0
        self.rate=100/(100/self.max_rate+pace*2)
        return arg0*n/(self.rate*1e6)
    def frames_done(self):return self.frames

class MeasurementTests(unittest.TestCase):
    def test_nanojoule_conversion(self):
        self.assertEqual(g.energy_nj(10,1),10)
        self.assertEqual(g.energy_nj(20,.5),40)
    def test_rate_below_target_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'RATE FAIL'):g.check_rate(.5,1)
    def test_calibrator_stops_on_insufficient_capacity(self):
        with self.assertRaisesRegex(RuntimeError,'maximum'):g.calibrate(MockReplay(.4),'FE_dense',100000,1,100e6,[])
    def test_calibrator_uses_measured_wait_cost(self):
        hist=[];pace,rate=g.calibrate(MockReplay(),'FE_gate',100000,1,100e6,hist)
        g.check_rate(rate,1);self.assertGreater(pace,0);self.assertGreater(len(hist),1)
    def test_wrong_image_rejected(self):
        ident={'magic':123,'build_id':456,'protocol':7,'default_replicas':4}
        with self.assertRaisesRegex(RuntimeError,'Wrong IP'):g.check_identity([0]*1024,ident)
    def test_pairing_and_replica_division(self):
        runs=[]
        for name,power in [('D0_empty',100),('FE_dense',140),('FE_gate',120),('FE_gate_iso',112),('FFT1024',130)]:
            runs.append({'mode':name,'mean_mW':power,'idle_bracket_mW':80,'rate_msps':1})
        d={'complete':True,'n_repl':4,'settings':{'quick':False,'target':1,'vectors':['v']},
           'repeats':[{'vector':'v','complete':True,'runs':[dict(x) for x in runs]} for _ in range(5)]}
        a=g.analyze(d)['vectors']['v']
        self.assertEqual(a['modes']['FE_gate']['D0_subtracted_nJ_per_replica']['mean'],5)
        self.assertEqual(a['paired_saving_nJ_per_replica']['dense_minus_gate']['mean'],5)
        self.assertEqual(a['paired_saving_nJ_per_replica']['dense_minus_gate']['n'],5)
        self.assertEqual(a['paired_saving_nJ_per_replica']['fft_minus_gate']['mean'],2.5)
        self.assertEqual(a['paired_saving_nJ_per_replica']['gate_minus_gate_iso']['mean'],2)
        self.assertEqual(a['paired_saving_nJ_per_replica']['dense_minus_gate_iso']['mean'],7)
        self.assertEqual(a['modes']['FE_gate_iso']['D0_subtracted_nJ_per_replica']['mean'],3)
    def test_old_four_mode_records_still_analyse(self):
        runs=[{'mode':k,'mean_mW':p,'idle_bracket_mW':80,'rate_msps':1} for k,p in [('D0_empty',100),('FE_dense',140),('FE_gate',120),('FFT1024',130)]]
        d={'complete':True,'n_repl':4,'settings':{'quick':False,'target':1,'vectors':['v']},
           'repeats':[{'vector':'v','complete':True,'runs':[dict(x) for x in runs]} for _ in range(3)]}
        a=g.analyze(d)['vectors']['v']
        self.assertNotIn('gate_minus_gate_iso',a['paired_saving_nJ_per_replica']);self.assertEqual(a['paired_saving_nJ_per_replica']['dense_minus_gate']['mean'],5)
    def test_v3_ablation_contrasts(self):
        runs=[{'mode':k,'mean_mW':p,'idle_bracket_mW':80,'rate_msps':1} for k,p in [('D0_empty',100),('FE_dense',140),('FE_dense_rom',132),
              ('FE_gate_iso',124),('FE_gate_iso_rom',116),('FE_v3',104),('FFT1024',108)]]
        d={'complete':True,'n_repl':4,'settings':{'quick':False,'target':1,'vectors':['v']},
           'repeats':[{'vector':'v','complete':True,'runs':[dict(x) for x in runs]} for _ in range(5)]}
        a=g.analyze(d)['vectors']['v'];ps=a['paired_saving_nJ_per_replica']
        self.assertEqual(a['modes']['FE_v3']['D0_subtracted_nJ_per_replica']['mean'],1)
        self.assertEqual(ps['dense_minus_v3']['mean'],9);self.assertEqual(ps['dense_minus_dense_rom']['mean'],2)
        self.assertEqual(ps['gate_iso_minus_gate_iso_rom']['mean'],2);self.assertEqual(ps['gate_iso_rom_minus_v3']['mean'],3)
        self.assertEqual(ps['fft_minus_v3']['mean'],1);self.assertNotIn('dense_minus_gate',ps)
        self.assertEqual(set(g.MODES),set(runs_k[0] for runs_k in [(r['mode'],) for r in runs]))
    def test_v2_records_with_gate_mode_still_analyse(self):
        runs=[{'mode':k,'mean_mW':p,'idle_bracket_mW':80,'rate_msps':1} for k,p in [('D0_empty',100),('FE_dense',140),('FE_gate',120),('FE_gate_iso',112),('FFT1024',130)]]
        d={'complete':True,'n_repl':4,'settings':{'quick':False,'target':1,'vectors':['v']},
           'repeats':[{'vector':'v','complete':True,'runs':[dict(x) for x in runs]} for _ in range(5)]}
        a=g.analyze(d)['vectors']['v'];self.assertEqual(a['paired_saving_nJ_per_replica']['gate_minus_gate_iso']['mean'],2)
    def test_one_repeat_no_ci(self):
        self.assertIsNone(g.interval([12])['ci95'])
    def test_ci_respects_between_repeat_variation(self):
        a=g.interval([-10,10,-10,10,0]);self.assertLess(a['ci95'][0],0);self.assertGreater(a['ci95'][1],0)

if __name__=='__main__':unittest.main()

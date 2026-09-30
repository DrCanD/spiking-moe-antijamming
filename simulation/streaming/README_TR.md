# Nedensel akış alıcısı

Alıcı 5 ms blokları tamponlar; her karar yalnız alınmış örneklere dayanır. ALE, delta kodlayıcı ve rezonatör durumları kod sözcükleri arasında korunur. Başarının ölçütü RS/CRC yük doğruluğudur.

Depo kökünden:

```bash
python simulation/streaming/verify_v5.py --output runs/verification.json
python simulation/streaming/run_v5.py --profile development --output runs/development --workers 2
python simulation/streaming/run_v5.py --profile validation --output runs/validation --selection runs/development/selection.json --workers 2
```

Geliştirme 720 akış, dondurulmuş seçimle bağımsız doğrulama 1.200 akış kullanır. Kaynak, ortam ve protokol kilitleri farklı koşuların karışmasını önler. Yeniden başlatma için aynı çıktı klasörü ve aynı kilit gerekir.

Ek karşılaştırmalar `run_comparison.py --study router`, `--study bands` ve `--study features` ile çalışır. `--smoke` kısa kontrol koşusudur. Sonuç klasörü `--output` ile seçilir. CPU yeterlidir; bu akış deneyleri GPU kullanmaz.

`frozen_v4/` sayısal çekirdeğin kayıtlı kaynak görüntüsüdür; sabit nokta donanım kontrolü `hardware/kv260/` altındadır.
